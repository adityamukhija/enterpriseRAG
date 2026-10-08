import google.generativeai as genai
from typing import List, Dict
from app.config import settings
from app.generation.prompts import get_prompt
from app.generation.router import ModelRouter
import time

genai.configure(api_key=settings.gemini_api_key)
router = ModelRouter()

class LLMGenerator:
    """Generate answers using Google Gemini (FREE)."""
    
    def generate(
        self,
        question: str,
        context_chunks: List[Dict],
        use_routing: bool = True,
    ) -> Dict:
        """
        THE CORE RAG GENERATION STEP:
        1. Format retrieved chunks into context string
        2. Route to appropriate model
        3. Call Gemini with context + question
        4. Return answer with metadata
        """
        # Step 1: Format context with source info
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", "N/A")
            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{chunk['content']}"
            )
        context_str = "\n\n---\n\n".join(context_parts)
        
        # Step 2: Choose model
        if use_routing:
            model_name = router.get_model(question)
        else:
            model_name = settings.default_model
        
        # Step 3: Call Gemini
        prompt = get_prompt("rag_answer")
        
        model = genai.GenerativeModel(
            model_name,
            system_instruction=prompt["system"],
        )
        
        start = time.time()
        response = model.generate_content(
            prompt["user"].format(context=context_str, question=question),
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temp for factual answers
                max_output_tokens=1000,
            ),
        )
        generation_time = (time.time() - start) * 1000
        
        answer = response.text
        
        return {
            "answer": answer,
            "model_used": model_name,
            "generation_time_ms": generation_time,
            "tokens_used": response.usage_metadata.total_token_count if response.usage_metadata else 0,
        }