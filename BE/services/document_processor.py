import aiofiles
import PyPDF2
import docx
import email
import os
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any
from fastapi import UploadFile
import magic
import tempfile

from models.schemas import DocumentInfo, DomainType

class DocumentProcessor:
    def __init__(self):
        self.upload_dir = os.getenv("UPLOAD_DIRECTORY", "uploads")
        self.metadata_file = "document_metadata.json"
        
        # Chunking configuration from environment
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.max_chunks = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "1000"))
        
        self.ensure_upload_directory()
    
    def ensure_upload_directory(self):
        """Ensure upload directory exists"""
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def process_document(self, file: UploadFile, domain: str) -> Dict[str, Any]:
        """
        Process uploaded document and extract text content
        """
        document_id = str(uuid.uuid4())
        file_path = os.path.join(self.upload_dir, f"{document_id}_{file.filename}")
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Extract text based on file type
        text_content = ""
        file_type = self._detect_file_type(file_path)
        
        if file_type == "pdf":
            text_content = self._extract_pdf_text(file_path)
        elif file_type in ["docx", "doc"]:
            text_content = self._extract_docx_text(file_path)
        elif file_type in ["eml", "msg"]:
            text_content = self._extract_email_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Chunk text for better processing
        text_chunks = self._chunk_text(text_content, self.chunk_size, self.chunk_overlap)
        
        # Limit chunks to prevent overwhelming the system
        if len(text_chunks) > self.max_chunks:
            print(f"Warning: Document has {len(text_chunks)} chunks, limiting to {self.max_chunks}")
            text_chunks = text_chunks[:self.max_chunks]
        
        # Save metadata
        document_info = DocumentInfo(
            document_id=document_id,
            filename=file.filename,
            file_type=file_type,
            domain=DomainType(domain),
            upload_date=datetime.now(),
            file_size=len(content),
            chunks_count=len(text_chunks),
            status="processed"
        )
        
        await self._save_document_metadata(document_info)
        
        return {
            "document_id": document_id,
            "text_chunks": text_chunks,
            "metadata": document_info.dict(),
            "file_path": file_path
        }
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type using python-magic"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type == "application/pdf":
                return "pdf"
            elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                return "docx"
            elif "message" in mime_type or file_path.endswith(('.eml', '.msg')):
                return "eml"
            else:
                # Fallback to extension
                return file_path.split('.')[-1].lower()
        except:
            return file_path.split('.')[-1].lower()
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
        return text
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error extracting DOCX text: {e}")
        return text
    
    def _extract_email_text(self, file_path: str) -> str:
        """Extract text from email file"""
        text = ""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                msg = email.message_from_file(file)
                
                # Extract subject and basic headers
                text += f"Subject: {msg.get('subject', '')}\n"
                text += f"From: {msg.get('from', '')}\n"
                text += f"To: {msg.get('to', '')}\n"
                text += f"Date: {msg.get('date', '')}\n\n"
                
                # Extract body
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                else:
                    text += msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error extracting email text: {e}")
        return text
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            
            chunk = text[start:end]
            chunks.append(chunk.strip())
            
            if end == text_length:
                break
            
            start = end - overlap
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    async def _save_document_metadata(self, document_info: DocumentInfo):
        """Save document metadata to JSON file"""
        try:
            metadata = {}
            if os.path.exists(self.metadata_file):
                async with aiofiles.open(self.metadata_file, 'r') as f:
                    content = await f.read()
                    if content.strip():
                        metadata = json.loads(content)
            
            metadata[document_info.document_id] = document_info.dict(default=str)
            
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(metadata, indent=2, default=str))
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    async def get_documents(self) -> List[DocumentInfo]:
        """Get list of all processed documents"""
        try:
            if not os.path.exists(self.metadata_file):
                return []
            
            async with aiofiles.open(self.metadata_file, 'r') as f:
                content = await f.read()
                if not content.strip():
                    return []
                
                metadata = json.loads(content)
                documents = []
                
                for doc_data in metadata.values():
                    documents.append(DocumentInfo(**doc_data))
                
                return documents
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete document and its metadata"""
        try:
            # Remove from metadata
            if os.path.exists(self.metadata_file):
                async with aiofiles.open(self.metadata_file, 'r') as f:
                    content = await f.read()
                    if content.strip():
                        metadata = json.loads(content)
                        if document_id in metadata:
                            del metadata[document_id]
                            
                            async with aiofiles.open(self.metadata_file, 'w') as f:
                                await f.write(json.dumps(metadata, indent=2, default=str))
            
            # Remove file (if exists)
            for file in os.listdir(self.upload_dir):
                if file.startswith(document_id):
                    os.remove(os.path.join(self.upload_dir, file))
                    break
            
            return {"status": "success", "document_id": document_id}
        except Exception as e:
            print(f"Error deleting document: {e}")
            return {"status": "error", "message": str(e)}
