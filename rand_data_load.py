# 示例：加载20 Newsgroups数据集并演示示例选取
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from rank_bm25 import BM25Okapi  # 安装：pip install rank_bm25
from sentence_transformers import SentenceTransformer, util  # 安装：pip install sentence-transformers

# 加载训练子集
train_data = fetch_20newsgroups(subset='train', categories=['sci.med','sci.space'], remove=('headers','footers','quotes'))
documents = train_data.data  # 文本列表
labels = train_data.target   # 对应标签

# 示例集准备：tokenize for BM25
tokenized_corpus = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# 以第一个文档作为查询例子，演示三种选取策略
query_text = documents[0]
k = 5

# 1. 随机选取
import random
random_examples = random.sample(documents, k)

# 2. BM25检索
scores = bm25.get_scores(query_text.split())
top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
bm25_examples = [documents[i] for i in top_idx]

# 3. 语义相似度选取
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model.encode(documents, convert_to_tensor=True)
query_embedding = model.encode([query_text], convert_to_tensor=True)
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
top_idx = cos_scores.argsort(descending=True)[:k]
semantic_examples = [documents[i] for i in top_idx]

print("\n\n\n\n\nRandom examples:", random_examples[:1])
print("BM25 examples:", bm25_examples[:1])
print("Semantic examples:", semantic_examples[:1])
