from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from app.models import QueryRequest, QueryResponse, SourceChunk
from app.database import init_db
from app.auth import create_token, verify_token
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.hybrid import HybridRetriever
from app.generation.llm import LLMGenerator
from app.cache.semantic_cache import SemanticCache
import uuid
import time
import tempfile
import os

app = FastAPI(title="Enterprise RAG Assistant", version="1.0.0")

# Initialize components
loader = DocumentLoader()
chunker = RecursiveChunker()
embedder = Embedder()
vector_store = VectorStore()
retriever = HybridRetriever()
generator = LLMGenerator()
cache = SemanticCache()

@app.on_event("startup")
def startup():
    init_db()
    vector_store.create_hnsw_index()
    print("RAG Assistant ready.")

@app.post("/auth/token")
def get_token(user_id: str = "default_user"):
    """Get a JWT token for API access."""
    token = create_token(user_id)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/ingest")
def ingest_document(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_token),
):
    """
    INGEST ENDPOINT — Upload a document to the RAG system.
    
    Flow: Upload → Load → Chunk → Embed → Store in pgvector
    """
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    
    # Save uploaded file temporarily
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = file.file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Step 1: Load document
        pages = loader.load(tmp_path)
        print(f"[{request_id}] Loaded {len(pages)} pages from {file.filename}")
        
        # Step 2: Chunk
        chunks = chunker.chunk(pages)
        # Override source name with original filename
        for chunk in chunks:
            chunk["source"] = file.filename
        print(f"[{request_id}] Split into {len(chunks)} chunks")
        
        # Step 3: Embed
        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed_texts(texts)
        print(f"[{request_id}] Generated {len(embeddings)} embeddings")
        
        # Step 4: Store
        vector_store.store_chunks(chunks, embeddings)
        
        # Rebuild BM25 index with new documents
        retriever.bm25_search.build_index()
        
        elapsed = (time.time() - start) * 1000
        
        return {
            "request_id": request_id,
            "filename": file.filename,
            "pages_loaded": len(pages),
            "chunks_created": len(chunks),
            "processing_time_ms": round(elapsed, 1),
            "message": f"Document '{file.filename}' ingested successfully.",
        }
    finally:
        os.unlink(tmp_path)

@app.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    user_id: str = Depends(verify_token),
):
    """
    QUERY ENDPOINT — Ask a question over ingested documents.
    
    Flow: Cache check → Retrieve (hybrid) → Generate (routed LLM) → Cache result → Return
    
    This is the MAIN RAG PIPELINE your resume describes.
    """
    request_id = str(uuid.uuid4())[:8]
    total_start = time.time()
    
    print(f"\n[{request_id}] Query: {request.question}")
    
    # Step 1: Check semantic cache
    if request.use_cache:
        cached = cache.get(request.question)
        if cached:
            return QueryResponse(
                answer=cached["answer"],
                sources=[SourceChunk(**s) for s in cached["sources"]],
                cached=True,
                model_used="cache",
                request_id=request_id,
                total_time_ms=round((time.time() - total_start) * 1000, 1),
            )
    
    # Step 2: Hybrid retrieval
    retrieval_start = time.time()
    retrieved_chunks = retriever.search(request.question, top_k=request.top_k)
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    if not retrieved_chunks:
        return QueryResponse(
            answer="No relevant documents found. Please ingest documents first.",
            sources=[],
            request_id=request_id,
        )
    
    print(f"[{request_id}] Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.1f}ms")
    
    # Step 3: Generate answer with routed LLM
    result = generator.generate(request.question, retrieved_chunks)
    
    # Build source citations
    sources = [
        SourceChunk(
            document_name=c["document_name"],
            page_number=c.get("page_number"),
            content=c["content"][:300] + "..." if len(c["content"]) > 300 else c["content"],
            relevance_score=round(c["relevance_score"], 4),
        )
        for c in retrieved_chunks
    ]
    
    # Step 4: Cache the result
    if request.use_cache:
        cache.set(
            request.question,
            result["answer"],
            [s.dict() for s in sources],
        )
    
    total_time = (time.time() - total_start) * 1000
    
    print(f"[{request_id}] Total: {total_time:.1f}ms (retrieval: {retrieval_time:.1f}ms, generation: {result['generation_time_ms']:.1f}ms)")
    
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        cached=False,
        model_used=result["model_used"],
        retrieval_time_ms=round(retrieval_time, 1),
        generation_time_ms=round(result["generation_time_ms"], 1),
        total_time_ms=round(total_time, 1),
        request_id=request_id,
    )

@app.delete("/cache")
def clear_cache(user_id: str = Depends(verify_token)):
    """Clear the semantic cache."""
    cache.clear()
    return {"message": "Cache cleared."}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Enterprise RAG Assistant"}