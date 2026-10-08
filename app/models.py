from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from app.database import Base
from app.config import settings

# ── SQLAlchemy ORM Models (Database Tables) ──

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_name = Column(String(500), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dimensions))
    page_number = Column(Integer)
    metadata_json = Column(Text)  # Store extra metadata as JSON
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Pydantic Models (API Request/Response) ──

class IngestRequest(BaseModel):
    """No body needed — file is uploaded via form-data."""
    pass

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_cache: Optional[bool] = True

class SourceChunk(BaseModel):
    document_name: str
    page_number: Optional[int]
    content: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    cached: bool = False
    model_used: str = ""
    retrieval_time_ms: float = 0
    generation_time_ms: float = 0
    total_time_ms: float = 0
    request_id: str = ""