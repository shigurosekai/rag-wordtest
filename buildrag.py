#背单词
#rag-chat 搭建向量数据库，然后生成例句，相似中文之类
#界面就用tkinter搭建，然后接语音api之类

import chromadb
from sentence_transformers import SentenceTransformer
import os


# ------------ 核心改动1：继承官方Embedding基类 ------------
class ChromaEmbedding:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
        self.dimension = self.model.get_sentence_embedding_dimension()  # 必须获取维度

    def __call__(self, input):                # ← must be named “input”
        # optional: ensure it’s a list
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input).tolist()


# ------------ 核心改动2：路径原生格式 + 维度注入 ------------
client = chromadb.PersistentClient(
    path=r"C:\Users\xujia\Desktop\affairs\wod\.model"  # 原生Windows路径
)

# 初始化模型（确保路径存在！）
embedder = ChromaEmbedding("D:/pythonw/models/paraphrase-miniLM-V2")

# 创建集合时传递维度到metadata（关键！）
collection = client.get_or_create_collection(
    name="rag64",
    embedding_function=embedder,
    metadata={
          # 将 dimension 改为 dim
        "hnsw:space": "cosine"  # 可选参数
    }
)


# ------------ 数据插入 ------------
def load_data(path):
    """
    从文件或文件夹加载数据
    :param path: 文件路径或文件夹路径
    :return: 文本内容列表
    """
    chunks = []
    
    if os.path.isfile(path):
        # 如果是单个文件
        with open(path, "r", encoding="utf-8") as f:
            chunks.extend([line.strip() for line in f if line.strip()])
    elif os.path.isdir(path):
        # 如果是文件夹，遍历所有.txt文件
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            chunks.extend([line.strip() for line in f if line.strip()])
                    except Exception as e:
                        print(f"警告：无法读取文件 {file_path}: {str(e)}")
    else:
        raise ValueError(f"路径不存在：{path}")
    
    return chunks

# 使用示例：可以传入文件或文件夹路径
chunks = load_data("rag-data")  # 假设您的数据文件夹名为 rag-data

# 批量插入（带ID）
for i in range(0, len(chunks), 5000):
    batch = chunks[i:i + 5000]
    collection.add(
        documents=batch,
        ids=[f"id_{i + j}" for j in range(len(batch))]
    )

print("✅ 数据插入完成！重启程序验证是否持久化")

def search(query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0]

# 示例搜索
query = "headquarters"
print(f"\n搜索结果：{search(query)}")


#api:sk-38c5057979a340579dc11feba7129aa3