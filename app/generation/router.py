import google.generativeai as genai
from app.config import settings
from app.generation.prompts import get_prompt

genai.configure(api_key=settings.gemini_api_key)

class ModelRouter:
    """
    Route queries to different model tiers based on complexity.
    
    YOUR RESUME SAYS: "budget LLMs for simple queries, flagship for complex"
    
    Simple queries (factual lookups) → gemini-2.0-flash (fast, free tier generous)
    Complex queries (reasoning, analysis) → gemini-2.0-pro (powerful, free tier smaller)
    
    This saved 62% on LLM costs in your resume.
    """
    
    def classify_complexity(self, question: str) -> str:
        """Use the fast model to classify query complexity."""
        prompt = get_prompt("query_complexity")
        
        model = genai.GenerativeModel(
            settings.default_model,
            system_instruction=prompt["system"],
        )
        
        response = model.generate_content(
            prompt["user"].format(question=question),
            generation_config=genai.GenerationConfig(
                max_output_tokens=10,
                temperature=0,
            ),
        )
        
        result = response.text.strip().upper()
        return result if result in ["SIMPLE", "COMPLEX"] else "SIMPLE"
    
    def get_model(self, question: str) -> str:
        """Return the appropriate model based on query complexity."""
        complexity = self.classify_complexity(question)
        
        if complexity == "COMPLEX":
            print(f"Query classified as COMPLEX → using {settings.flagship_model}")
            return settings.flagship_model
        else:
            print(f"Query classified as SIMPLE → using {settings.default_model}")
            return settings.default_model