"""
PDF text extraction using PyMuPDF
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional


class PDFProcessor:
    """Extract text from PDF certificates"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> str:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                text_content.append(text)
            
            doc.close()
            
            # Join all pages with newlines
            full_text = "\n\n".join(text_content)
            return full_text.strip()
            
        except Exception as e:
            print(f"✗ Error extracting text from {pdf_path.name}: {e}")
            return ""
    
    @staticmethod
    def process_certificates(certificate_files: List[str], certificates_dir: Path) -> Dict[str, str]:
        """
        Process multiple certificate PDFs
        
        Args:
            certificate_files: List of certificate filenames
            certificates_dir: Directory containing certificates
            
        Returns:
            Dictionary mapping filename to extracted text
        """
        results = {}
        
        for cert_file in certificate_files:
            cert_path = certificates_dir / cert_file
            
            if not cert_path.exists():
                print(f"⚠ Certificate not found: {cert_file}")
                results[cert_file] = ""
                continue
            
            text = PDFProcessor.extract_text_from_pdf(cert_path)
            results[cert_file] = text
            
            if text:
                print(f"✓ Extracted {len(text)} chars from {cert_file}")
            else:
                print(f"⚠ No text extracted from {cert_file}")
        
        return results
    
    @staticmethod
    def combine_certificate_texts(cert_texts: Dict[str, str]) -> str:
        """
        Combine all certificate texts into single string for embedding
        
        Args:
            cert_texts: Dictionary of filename -> text
            
        Returns:
            Combined text with certificate names as headers
        """
        combined = []
        
        for filename, text in cert_texts.items():
            if text:
                # Add filename as header
                cert_name = filename.replace('.pdf', '').replace('-', ' ').title()
                combined.append(f"=== {cert_name} ===")
                combined.append(text)
                combined.append("")  # Empty line separator
        
        return "\n".join(combined)