from rank_bm25 import BM25Okapi
from typing import List, Dict
from app.database import SessionLocal
from app.models import DocumentChunk

class BM25Search:
    """Traditional keyword search using BM25 algorithm."""
    
    def __init__(self):
        self.corpus = []
        self.chunks = []
        self.bm25 = None
    
    def build_index(self):
        """Load all chunks and build BM25 index."""
        db = SessionLocal()
        try:
            all_chunks = db.query(DocumentChunk).all()
            self.chunks = [
                {
                    "document_name": c.document_name,
                    "content": c.content,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in all_chunks
            ]
            # BM25 needs tokenized corpus (list of word lists)
            self.corpus = [chunk["content"].lower().split() for chunk in self.chunks]
            self.bm25 = BM25Okapi(self.corpus)
            print(f"BM25 index built with {len(self.corpus)} documents.")
        finally:
            db.close()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using BM25 keyword matching."""
        if not self.bm25:
            self.build_index()
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                chunk = self.chunks[idx].copy()
                chunk["relevance_score"] = float(scores[idx])
                results.append(chunk)
        
        return results