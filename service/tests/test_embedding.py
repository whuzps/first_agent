from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 会自动下载小型模型
embeddings = model.encode(["Hello from Python 3.12!"])
print(embeddings.shape)  # 输出类似 (1, 384)