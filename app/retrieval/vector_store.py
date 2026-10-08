from sqlalchemy import text
from typing import List, Dict, Optional
from app.database import SessionLocal
from app.models import DocumentChunk
from app.ingestion.embedder import Embedder
from app.config import settings
import json
import time

embedder = Embedder()

class VectorStore:
    """Store and search embeddings in PostgreSQL with pgvector."""
    
    def store_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Store document chunks with their embeddings in PostgreSQL.
        
        This is the INGESTION side of RAG — happens once per document.
        """
        db = SessionLocal()
        try:
            for chunk, embedding in zip(chunks, embeddings):
                doc_chunk = DocumentChunk(
                    document_name=chunk["source"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    embedding=embedding,
                    page_number=chunk.get("page_number"),
                    metadata_json=json.dumps({
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                    }),
                )
                db.add(doc_chunk)
            db.commit()
            print(f"Stored {len(chunks)} chunks in vector store.")
        finally:
            db.close()
    
    def search(self, query: str, top_k: int = settings.top_k) -> List[Dict]:
        """
        VECTOR SEARCH — The core of RAG retrieval.
        
        How it works:
        1. Convert the user's question into an embedding
        2. Use pgvector's <=> operator to find closest embeddings
        3. <=> calculates cosine distance (lower = more similar)
        4. Return top_k most similar chunks
        
        The HNSW index makes this fast even with millions of chunks.
        Without the index: O(n) linear scan — checks every row
        With HNSW index: O(log n) approximate search — much faster
        """
        query_embedding = embedder.embed_query(query)
        
        db = SessionLocal()
        try:
            start = time.time()
            
            # pgvector cosine distance search
            # <=> is the cosine distance operator
            # 1 - distance = similarity (we return this as relevance_score)
            results = db.execute(
                text("""
                    SELECT 
                        id, document_name, chunk_index, content, 
                        page_number, metadata_json,
                        1 - (embedding <=> :query_embedding) AS relevance_score
                    FROM document_chunks
                    ORDER BY embedding <=> :query_embedding
                    LIMIT :top_k
                """),
                {
                    "query_embedding": str(query_embedding),
                    "top_k": top_k,
                },
            ).fetchall()
            
            elapsed_ms = (time.time() - start) * 1000
            
            chunks = []
            for row in results:
                chunks.append({
                    "document_name": row.document_name,
                    "content": row.content,
                    "page_number": row.page_number,
                    "relevance_score": float(row.relevance_score),
                    "chunk_index": row.chunk_index,
                })
            
            print(f"Vector search returned {len(chunks)} results in {elapsed_ms:.1f}ms")
            return chunks
        finally:
            db.close()
    
    def create_hnsw_index(self):
        """
        Create HNSW index for fast approximate nearest neighbor search.
        
        HNSW = Hierarchical Navigable Small World
        
        Think of it like a skip list for vectors:
        - Without index: compare query against EVERY embedding (slow)
        - With HNSW: navigate a graph structure to find approximate nearest 
          neighbors in O(log n) time
        
        m = 16: each node connects to 16 neighbors (higher = more accurate, more memory)
        ef_construction = 64: how many candidates to consider during build (higher = slower build, better quality)
        
        YOUR RESUME SAYS THIS. Know what these parameters mean.
        """
        db = SessionLocal()
        try:
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
                ON document_chunks 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            db.commit()
            print("HNSW index created.")
        finally:
            db.close()