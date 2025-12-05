# kpi_predictor.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd

def load_baselines(path="data/sample_baselines.csv"):
    df = pd.read_csv(path)
    return df['baseline'].tolist()

def load_messages(path="data/sample_messages.csv"):
    df = pd.read_csv(path)
    return df['message'].tolist()

def embed(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts)

def build_index(baseline_embeddings):
    dimension = baseline_embeddings.shape[1]
    index = faiss.index_factory(dimension, "HNSW32")
    index.hnsw.efConstruction = 200
    index.add(np.array(baseline_embeddings).astype("float32"))
    return index

def predict_kpi(message, baselines, index, model):
    embedding = model.encode([message])
    D, I = index.search(np.array(embedding).astype("float32"), k=1)
    return baselines[I[0][0]]

if __name__ == "__main__":
    baselines = load_baselines()
    messages = load_messages()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    baseline_embeddings = model.encode(baselines)
    index = build_index(baseline_embeddings)

    for msg in messages:
        kpi = predict_kpi(msg, baselines, index, model)
        print(f"Message: {msg}\nPredicted KPI: {kpi}\n")
