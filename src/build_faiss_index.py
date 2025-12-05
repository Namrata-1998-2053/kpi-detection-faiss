# build_faiss_index.py

import faiss
import numpy as np

def build_faiss_hnsw(embeddings, M=32, efC=200):
    dimension = embeddings.shape[1]
    index = faiss.index_factory(dimension, "HNSW32")
    index.hnsw.efConstruction = efC
    index.add(np.array(embeddings).astype("float32"))
    return index

if __name__ == "__main__":
    dummy_embeddings = np.random.rand(5, 384).astype("float32")
    index = build_faiss_hnsw(dummy_embeddings)
    print("FAISS HNSW index created with 5 dummy embeddings")
