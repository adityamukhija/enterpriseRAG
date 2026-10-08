from typing import List, Dict
from app.config import settings

class RecursiveChunker:
    """
    Split text into chunks using recursive character splitting.
    
    How it works:
    1. Try splitting on double newlines (paragraphs)
    2. If chunks are still too big, split on single newlines
    3. If still too big, split on sentences (periods)
    4. If still too big, split on spaces (words)
    5. Last resort: split on characters
    
    This preserves natural text boundaries as much as possible.
    """
    
    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, pages: List[Dict]) -> List[Dict]:
        """
        Takes loader output, returns chunks with metadata.
        Each chunk = {content, page_number, source, chunk_index}
        """
        all_chunks = []
        chunk_index = 0
        
        for page in pages:
            text = page["text"]
            splits = self._recursive_split(text, self.separators)
            
            # Merge small splits into chunks of target size with overlap
            merged = self._merge_with_overlap(splits)
            
            for chunk_text in merged:
                all_chunks.append({
                    "content": chunk_text,
                    "page_number": page.get("page_number"),
                    "source": page.get("source", "unknown"),
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
        
        return all_chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the best separator available."""
        if len(text) <= self.chunk_size:
            return [text]
        
        # Find the best separator (first one that exists in text)
        best_sep = separators[-1]  # Default: character split
        for sep in separators:
            if sep in text:
                best_sep = sep
                break
        
        if best_sep == "":
            # Character-level split (last resort)
            parts = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        else:
            parts = text.split(best_sep)
        
        # Recursively split any part that's still too big
        result = []
        remaining_seps = separators[separators.index(best_sep) + 1:] if best_sep in separators else separators[-1:]
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._recursive_split(part, remaining_seps))
        
        return result
    
    def _merge_with_overlap(self, splits: List[str]) -> List[str]:
        """
        Merge small splits into chunks of ~chunk_size with overlap.
        
        THIS IS THE KEY CONCEPT:
        Overlap means the last N characters of chunk 1 appear at the 
        start of chunk 2. This prevents losing context at boundaries.
        
        Example with chunk_size=100, overlap=20:
          Chunk 1: "The hotel is located in Dubai. It has 200 rooms and..."
          Chunk 2: "200 rooms and a rooftop pool. The restaurant serves..."
          
        "200 rooms and" appears in both chunks. If someone asks about 
        the hotel's pool, chunk 2 has full context about the rooms AND pool.
        """
        chunks = []
        current = ""
        
        for split in splits:
            # If adding this split would exceed chunk_size, save current and start new
            if current and len(current) + len(split) + 1 > self.chunk_size:
                chunks.append(current.strip())
                
                # Start new chunk with overlap from end of previous
                if self.chunk_overlap > 0:
                    overlap_text = current[-self.chunk_overlap:]
                    current = overlap_text + " " + split
                else:
                    current = split
            else:
                current = (current + " " + split).strip() if current else split
        
        # Don't forget the last chunk
        if current.strip():
            chunks.append(current.strip())
        
        return chunks