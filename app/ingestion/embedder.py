from google import genai
from typing import List
from app.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

class Embedder:
    """
    Generate embeddings using Google Gemini (FREE, API-based).
    No model download needed.
    """
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        all_embeddings = []
        for text in texts:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
            )
            all_embeddings.append(result.embeddings[0].values)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]