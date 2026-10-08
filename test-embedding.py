# test_embedding.py
from app.ingestion.embedder import Embedder
import numpy as np

embedder = Embedder()

# Embed three sentences
texts = [
    "The hotel has a swimming pool",
    "The resort features an outdoor pool area",
    "The flight departs at 6am from terminal 3",
]

embeddings = embedder.embed_texts(texts)

print(f"Each embedding has {len(embeddings[0])} dimensions (384 for MiniLM)\n")

# Calculate cosine similarity
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"'pool' vs 'pool area':  {cosine_sim(embeddings[0], embeddings[1]):.4f}")  # Should be HIGH (>0.85)
print(f"'pool' vs 'flight':     {cosine_sim(embeddings[0], embeddings[2]):.4f}")  # Should be LOW (<0.75)
print(f"'pool area' vs 'flight': {cosine_sim(embeddings[1], embeddings[2]):.4f}") # Should be LOW (<0.75)