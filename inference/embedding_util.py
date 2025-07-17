from sentence_transformers import SentenceTransformer
import numpy as np

# 임베딩 모델 로드 (한 번만)
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

def encode_texts(texts):
    return model.encode(texts, convert_to_numpy=True)

def save_embeddings(embeddings, path):
    np.save(path, embeddings)

def load_embeddings(path):
    return np.load(path)