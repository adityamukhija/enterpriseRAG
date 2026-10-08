from typing import List, Dict
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Search

class HybridRetriever:
    """
    Combine dense (vector) and sparse (BM25) retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    
    RRF formula: score = sum(1 / (k + rank)) for each result list
    where k = 60 (constant to prevent high-ranked items from dominating)
    
    Why RRF over weighted sum?
    - Doesn't require normalizing scores across different systems
    - Works well even when score distributions are very different
    - Simple, no hyperparameters to tune (besides k)
    """
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_search = BM25Search()
        self.rrf_k = 60  # Standard RRF constant
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search using RRF to combine vector + BM25 results."""
        
        # Get results from both retrievers
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        bm25_results = self.bm25_search.search(query, top_k=top_k * 2)
        
        # Build RRF scores
        # Key = chunk content (use as unique identifier)
        rrf_scores = {}
        chunk_data = {}
        
        # Score from vector search
        for rank, chunk in enumerate(vector_results):
            key = chunk["content"][:100]  # Use first 100 chars as key
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_data[key] = chunk
        
        # Score from BM25 search
        for rank, chunk in enumerate(bm25_results):
            key = chunk["content"][:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in chunk_data:
                chunk_data[key] = chunk
        
        # Sort by combined RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        results = []
        for key in sorted_keys[:top_k]:
            chunk = chunk_data[key]
            chunk["relevance_score"] = rrf_scores[key]
            results.append(chunk)
        
        return results