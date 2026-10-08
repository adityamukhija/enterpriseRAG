from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict

class DocumentLoader:
    """Load text from various file formats."""
    
    def load(self, file_path: str) -> List[Dict]:
        """
        Returns list of {text: str, page_number: int, source: str}
        Each item = one page or section of the document.
        """
        path = Path(file_path)
        
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        elif path.suffix.lower() in [".txt", ".md"]:
            return self._load_text(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def _load_pdf(self, path: Path) -> List[Dict]:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page_number": i + 1,
                    "source": path.name,
                })
        return pages
    
    def _load_text(self, path: Path) -> List[Dict]:
        text = path.read_text(encoding="utf-8")
        return [{
            "text": text,
            "page_number": 1,
            "source": path.name,
        }]