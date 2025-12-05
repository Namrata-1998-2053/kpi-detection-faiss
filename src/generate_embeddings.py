# generate_embeddings.py

from sentence_transformers import SentenceTransformer
import pandas as pd

def load_baselines(file_path):
    df = pd.read_csv(file_path)
    return df['baseline'].tolist()

def generate_embeddings(baselines):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(baselines)
    return embeddings

if __name__ == "__main__":
    baselines = load_baselines("data/sample_baselines.csv")
    print("Loaded baselines:", baselines)

    embeddings = generate_embeddings(baselines)
    print("Generated embeddings shape:", embeddings.shape)
