from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import re
import random
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

app = Flask(__name__)
# 请务必设置一个安全的密钥，用于会话管理。在生产环境中，这应该是一个复杂的随机字符串。
app.config['SECRET_KEY'] = 'your_super_secret_key_for_scoring_feature_replace_me'  # 请替换为一个复杂的随机字符串！
app.secret_key = 'your_secret_key'

# 初始化数据库和模型
embedding_model = SentenceTransformer("D:/pythonw/models/paraphrase-miniLM-V2")
chroma_client = chromadb.PersistentClient(path=r"C:\Users\xujia\Desktop\affairs\wod\.model")
try:
    collection = chroma_client.get_or_create_collection("rag64")
except Exception as e:
    print(f"Error getting/creating collection 'rag64': {e}")
    # 紧急退出，避免后续错误，但为了保持服务运行，此处不退出，而是设置collection为None
    collection = None


# 会话初始化：在每次请求之前，检查并初始化分数和尝试次数
@app.before_request
def before_request():
    if 'score' not in session:
        session['score'] = 0
    if 'total_attempts' not in session:
        session['total_attempts'] = 0


def fetch_word(text):
    """
    从文本中提取英文单词，例如 'apple /æpl/ n. 苹果' -> 'apple'
    此函数严格只保留ASCII英文。
    """
    if not text:
        return None

    match = re.search(r'^[a-zA-Z]+', text)
    if match:
        return match.group(0)

    parts = text.split(' ')
    if parts:
        english = ''.join(c for c in parts[0].strip() if 'a' <= c.lower() <= 'z')
        return english if english else None

    return None


def fetch_trans(text):
    """
    从文本中提取中文释义，例如 'apple /æpl/ n. 苹果' -> '苹果'
    此函数能更准确地提取中文。
    """
    if not text:
        return None

    match = re.search(r'([\u4e00-\u9fff]+[^.，,;；！？。]*)$', text)
    if match:
        chinese = ''.join(c for c in match.group(1) if '\u4e00' <= c <= '\u9fff')
        return chinese if chinese else None

    all_chinese_matches = re.findall(r'[\u4e00-\u9fff]+', text)
    if all_chinese_matches:
        return all_chinese_matches[-1]

    return None


def generate_example_sentence(english_word, chinese_trans):
    """
    为给定的英文单词和中文翻译生成例句
    """
    # 简单的例句模板
    example_templates = [
        f"The word '{english_word}' means '{chinese_trans}' in Chinese.",
        f"'{english_word}' is translated as '{chinese_trans}'.",
        f"In English, we say '{english_word}' for '{chinese_trans}'.",
        f"The English word '{english_word}' corresponds to '{chinese_trans}' in Chinese.",
        f"'{english_word}' can be understood as '{chinese_trans}'."
    ]
    
    return random.choice(example_templates)


def get_similar_words(text, top_k=4):
    """输入一段文本，返回数据库中最相关的top_k条内容（包括自身）"""
    if embedding_model is None or collection is None:
        print("[get_similar_words] Warning: Embedding model or ChromaDB collection not initialized.")
        return []
    try:
        embedding = embedding_model.encode(text).tolist()
        results = collection.query(query_embeddings=[embedding], n_results=top_k)
        return results['documents'][0] if 'documents' in results else []
    except Exception as e:
        print(f"[get_similar_words] Error querying similar words: {e}")
        return []


def get_random_word(seed=114514):
    """
    用随机种子从数据库中随机取出一个有效英文单词的原始条目。
    此版本通过获取所有文档并过滤，确保返回的是有效英文条目。
    """
    random.seed(seed)

    if collection is None:
        print("[get_random_word] Error: ChromaDB collection not initialized.")
        return None

    try:
        all_docs = collection.get(limit=100000)  # 获取所有文档，或者设置一个足够大的限制
        words = all_docs['documents'] if 'documents' in all_docs and all_docs['documents'] else []
    except Exception as e:
        print(f"Debug - get_random_word: Error fetching all documents from collection: {e}")
        return None

    if not words:
        print("Debug - get_random_word: No words found in database after collection.get().")
        return None

    valid_english_entries = []
    for entry in words:
        if entry and entry.strip():
            english_part = fetch_word(entry)
            chinese_part = fetch_trans(entry)

            is_pure_ascii_alpha_word = True
            if english_part:
                for char_c in english_part:
                    if not (('a' <= char_c.lower() <= 'z') or char_c.isspace()):
                        is_pure_ascii_alpha_word = False
                        break
            else:
                is_pure_ascii_alpha_word = False

            if english_part and chinese_part and is_pure_ascii_alpha_word:
                valid_english_entries.append(entry)

    if not valid_english_entries:
        print("Debug - get_random_word: No valid English entries found in database after strict filtering.")
        return None

    selected_word_entry = random.choice(valid_english_entries)
    return selected_word_entry


@app.route('/')
def index():
    print("[Route /] Serving index.html")
    return render_template('index.html')


@app.route("/get_word", methods=["POST"])
def get_word():
    print("\n--- Processing new /get_word request ---")
    data = request.get_json()
    game_mode = data.get('mode', 'en-to-zh')  # 默认为英文猜中文模式

    seed = random.randint(0, 999999)
    word_from_db_raw = get_random_word(seed=seed)

    if not word_from_db_raw:
        print("Debug - get_word: No valid word from database after random selection, returning error message.")
        return jsonify({
            "base_word": "error",
            "options": ["请重试", "点击按钮", "刷新页面", "联系支持"],
            "answer": "请重试"
        })

    # 提取英文单词和中文翻译
    english_word = fetch_word(word_from_db_raw)
    chinese_trans = fetch_trans(word_from_db_raw)

    if not english_word or not chinese_trans:
        print(f"Debug - get_word: Failed to extract word/translation from: '{word_from_db_raw}'")
        return jsonify({
            "base_word": "error",
            "options": ["请重试", "点击按钮", "刷新页面", "联系支持"],
            "answer": "请重试"
        })

    # 根据游戏模式设置问题和选项
    if game_mode == 'en-to-zh':
        base_word = english_word
        correct_answer = chinese_trans
        # 获取相似词的中文翻译作为选项
        related_words = get_similar_words(f"please give an English word similar to '{english_word}',seed={seed}",
                                          top_k=10)
        options = [chinese_trans]  # 确保正确答案在选项中
        for raw_word in related_words:
            if len(options) >= 4:
                break
            if raw_word != word_from_db_raw:
                trans = fetch_trans(raw_word)
                if trans and trans != chinese_trans and trans not in options:
                    options.append(trans)
    else:  # zh-to-en 模式
        base_word = chinese_trans
        correct_answer = english_word
        # 获取相似词作为选项
        related_words = get_similar_words(f"please give an English word similar to '{english_word}',seed={seed}",
                                          top_k=10)
        options = [english_word]  # 确保正确答案在选项中
        for raw_word in related_words:
            if len(options) >= 4:
                break
            if raw_word != word_from_db_raw:
                eng = fetch_word(raw_word)
                if eng and eng != english_word and eng not in options:
                    options.append(eng)

    # 如果选项不足4个，填充"未知选项"
    while len(options) < 4:
        options.append("未知选项" if game_mode == 'en-to-zh' else "unknown")

    # 裁剪到只剩4个选项并打乱顺序
    options = options[:4]
    random.shuffle(options)

    # 生成例句
    example_sentence = generate_example_sentence(english_word, chinese_trans)

    result = {
        "base_word": base_word,
        "options": options,
        "answer": correct_answer,
        "example_sentence": example_sentence
    }
    print(f"Debug - Final result payload sent to frontend: {result}")
    return jsonify(result)


# 新增的路由：处理答案提交和计分
@app.route('/check_answer', methods=['POST'])
def check_answer():
    print("--- Processing new /check_answer request (POST) ---")
    data = request.get_json()
    user_answer = data.get('user_answer')
    correct_option = data.get('correct_option')
    action = data.get('action')  # 获取前端的 action

    # 如果是前端仅仅请求获取分数，不进行计分和尝试次数的修改
    if action == 'get_score_only':
        correctness_rate = (session['score'] / session['total_attempts'] * 100) if session['total_attempts'] > 0 else 0
        return jsonify({
            'score': session['score'],
            'total_attempts': session['total_attempts'],
            'correctness_rate': round(correctness_rate, 2)
        })

    # 以下是正常提交答案的逻辑
    session['total_attempts'] += 1
    is_correct = (user_answer == correct_option)

    # 获取 base_word，用于记录错题
    base_word = data.get('base_word')

    if is_correct:
        session['score'] += 1
    else:
        # 如果答错了，将错题信息添加到 session 中
        wrong_words = session.get('wrong_words', [])
        wrong_words.append({
            'base_word': base_word,  # 记录问题单词
            'correct_answer': correct_option  # 记录正确答案
        })
        session['wrong_words'] = wrong_words  # 更新 session

    correctness_rate = (session['score'] / session['total_attempts'] * 100) if session['total_attempts'] > 0 else 0

    print(f"[check_answer] User answer: {user_answer}, Correct option: {correct_option}, Is correct: {is_correct}")
    print(
        f"[check_answer] Score: {session['score']}, Total Attempts: {session['total_attempts']}, Rate: {correctness_rate:.2f}%")

    return jsonify({
        'is_correct': is_correct,
        'score': session['score'],
        'total_attempts': session['total_attempts'],
        'correctness_rate': round(correctness_rate, 2)
    })


# 添加一个临时的辅助路由，用于在服务器启动后手动添加测试单词
# 请注意：这个路由不是游戏核心功能的一部分，只是为了方便测试和演示
# 如果数据库已经有数据，可以不运行此路由
@app.route('/add_test_words_for_rag64')
def add_test_words_for_rag64():
    print("--- Processing new /add_test_words_for_rag64 request ---")

    test_words_data = [
        "emperor /ˈempərə(r)/ n. 皇帝；君主",
        "slope /sləʊp/ n. 斜坡；倾斜；斜率 vi. 倾斜；有坡度 vt. 使倾斜",
        "recourse /rɪˈkɔːs/ n. 求助；追索权",
        "facial /ˈfeɪʃl/ adj. 面部的；脸的 n. 面部美容",
        "diversity /daɪˈdɜːsəti/ n. 多样性；差异；少数民族",
        "harmony /ˈhɑːrməni/ n. 和谐；融洽；和声",
        "innovate /ˈɪnəveɪt/ v. 创新；改革",
        "serenity /səˈrɛnəti/ n. 宁静；晴朗",
        "ubiquitous /juːˈbɪkwɪtəs/ adj. 普遍存在的；无处不在的",
        "zealous /ˈzɛləs/ adj. 热情的；积极的"
    ]

    documents = []
    metadatas = []
    ids = []

    try:
        # 获取当前数据库中已有的 ID，防止重复添加
        if collection:
            existing_ids_result = collection.peek(limit=collection.count())
            existing_ids = set(existing_ids_result.get('ids', []))
        else:
            existing_ids = set()
            print("[add_test_words_for_rag64] Warning: Collection not initialized, cannot check existing IDs.")

        new_words_count = 0
        for i, word_entry in enumerate(test_words_data):
            # 创建一个稳定的ID，例如基于内容的哈希或简单的索引
            # 由于是测试数据，我们可以使用简单的递增ID
            word_id = f"test_word_rag64_{i}"
            if word_id not in existing_ids:
                documents.append(word_entry)
                metadatas.append({"source": "manual_test_data"})
                ids.append(word_id)
                new_words_count += 1

        if collection is None:
            return "ChromaDB collection not initialized. Cannot add words.", 500

        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[add_test_words_for_rag64] Added {new_words_count} new test words.")
        else:
            print("[add_test_words_for_rag64] No new words to add or words already exist.")

        current_total = collection.count()
        print(f"[add_test_words_for_rag64] Current total words in database: {current_total}")
        return f"已成功添加 {new_words_count} 个测试单词到数据库！当前数据库共有 {current_total} 个单词。"

    except Exception as e:
        print(f"[add_test_words_for_rag64] Failed to add words to ChromaDB: {e}")
        return f"添加单词失败：{e}", 500


@app.route('/get_wrong_words', methods=['GET'])
def get_wrong_words():
    print("--- Processing new /get_wrong_words request ---")
    wrong_words = session.get('wrong_words', [])
    return jsonify({'wrong_words': wrong_words})


@app.route('/clear_wrong_words', methods=['POST'])
def clear_wrong_words():
    print("--- Processing new /clear_wrong_words request ---")
    session['wrong_words'] = []
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
