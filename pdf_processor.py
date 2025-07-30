import fitz  # PyMuPDF
from typing import List, Dict

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_text_chunks(self, chunk_size: int = 1000) -> List[Dict]:
        """
        Extract text from PDF and create chunks with metadata.
        
        Args:
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        chunks = []
        doc = fitz.open(self.pdf_path)
        
        current_section = None
        current_subsection = None
        current_chunk = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("dict")
            
            for block in text.get("blocks", []):
                if "lines" in block:
                    block_text = " ".join([" ".join([span["text"] for span in line["spans"]]) 
                                         for line in block["lines"] if "spans" in line])
                    
                    if not block_text.strip():
                        continue
                        
                    # Detect section headers (bold text with numbers)
                    is_section = False
                    is_subsection = False
                    
                    # Get font info
                    font = block.get("lines", [{}])[0].get("spans", [{}])[0].get("font", "") if block.get("lines") else ""
                    font_size = block.get("lines", [{}])[0].get("spans", [{}])[0].get("size", 0) if block.get("lines") else 0
                    
                    # Check if this is a section header (bold text with number)
                    if font_size > 12:  # Larger font size indicates header
                        if any(char.isdigit() for char in block_text):
                            is_section = True
                            current_section = block_text.strip()
                            current_subsection = None
                        elif "✅" in block_text:  # Bullet points indicate subsection
                            is_subsection = True
                            current_subsection = block_text.strip()
                    
                    # Add metadata about headers
                    chunk_metadata = {
                        'page': page_num + 1,
                        'section': current_section,
                        'subsection': current_subsection,
                        'is_header': is_section,
                        'is_subsection': is_subsection,
                        'font': font,
                        'font_size': font_size
                    }
                    
                    # Add to current chunk
                    if len(current_chunk) + len(block_text) <= chunk_size:
                        current_chunk += block_text + " "
                    else:
                        # Save current chunk and start new one
                        if current_chunk.strip():
                            chunks.append({
                                'text': current_chunk.strip(),
                                'source': 'PDF',
                                'metadata': [chunk_metadata]
                            })
                        current_chunk = block_text + " "
            
            # Add last chunk for the page
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': 'PDF',
                    'metadata': [chunk_metadata]
                })
                current_chunk = ""
        
        doc.close()
        return chunks
