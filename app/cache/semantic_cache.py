import redis
import json
import numpy as np
from typing import Optional, Dict
from app.config import settings
from app.ingestion.embedder import Embedder
import time

embedder = Embedder()

class SemanticCache:
    """
    Redis-backed semantic cache for RAG queries.
    
    YOUR RESUME SAYS THIS. Here's how it works:
    
    1. User asks "What is the cancellation policy?"
    2. We embed this question → vector
    3. Check Redis: are there any cached questions with similar embeddings?
    4. If cosine_similarity > 0.95 → return cached answer (skip LLM entirely!)
    5. If not → call LLM, cache the question embedding + answer
    
    This saves massive LLM costs because users often ask similar questions.
    "What's the cancellation policy?" and "cancellation rules?" are different 
    strings but the SAME question semantically.
    
    Your resume claims 55% token savings from this.
    """
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.threshold = settings.cache_similarity_threshold
        self.ttl = settings.cache_ttl_seconds
        self.cache_prefix = "rag_cache:"
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get(self, question: str) -> Optional[Dict]:
        """
        Check if a semantically similar question has been cached.
        
        We scan ALL cached embeddings and find the most similar one.
        If similarity > threshold, return the cached answer.
        
        For production at scale, you'd use a vector index in Redis 
        (RediSearch) instead of scanning. But this works for learning.
        """
        query_embedding = embedder.embed_query(question)
        
        # Get all cached keys
        keys = self.redis_client.keys(f"{self.cache_prefix}*")
        
        best_match = None
        best_score = 0
        
        for key in keys:
            cached = json.loads(self.redis_client.get(key))
            similarity = self._cosine_similarity(query_embedding, cached["embedding"])
            
            if similarity > best_score:
                best_score = similarity
                best_match = cached
        
        if best_match and best_score >= self.threshold:
            print(f"Cache HIT! Similarity: {best_score:.4f} (threshold: {self.threshold})")
            return {
                "answer": best_match["answer"],
                "sources": best_match.get("sources", []),
                "cached": True,
                "cache_similarity": best_score,
            }
        
        if best_score > 0:
            print(f"Cache MISS. Best similarity: {best_score:.4f} (threshold: {self.threshold})")
        else:
            print("Cache MISS. No cached queries found.")
        
        return None
    
    def set(self, question: str, answer: str, sources: list):
        """Cache a question-answer pair with the question's embedding."""
        embedding = embedder.embed_query(question)
        
        cache_data = {
            "question": question,
            "embedding": embedding,
            "answer": answer,
            "sources": sources,
            "timestamp": time.time(),
        }
        
        # Use question hash as key
        key = f"{self.cache_prefix}{hash(question)}"
        self.redis_client.setex(
            key,
            self.ttl,
            json.dumps(cache_data),
        )
        print(f"Cached answer for: '{question[:50]}...'")
    
    def clear(self):
        """Clear all cached entries."""
        keys = self.redis_client.keys(f"{self.cache_prefix}*")
        if keys:
            self.redis_client.delete(*keys)
        print("Cache cleared.")