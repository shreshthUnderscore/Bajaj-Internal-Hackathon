import aiohttp
import asyncio
import tempfile
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import aiofiles

from services.document_processor import DocumentProcessor

class URLDocumentProcessor:
    def __init__(self):
        self.document_processor = DocumentProcessor()
    
    async def download_and_process_document(self, url: str, domain: str = "insurance") -> Dict[str, Any]:
        """
        Download document from URL and process it
        """
        try:
            # Download the document
            file_path, filename = await self._download_document(url)
            
            # Create a mock UploadFile object
            mock_file = await self._create_mock_upload_file(file_path, filename)
            
            # Process the document
            result = await self.document_processor.process_document(mock_file, domain)
            
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return result
            
        except Exception as e:
            print(f"Error processing URL document: {e}")
            raise Exception(f"Failed to process document from URL: {str(e)}")
    
    async def _download_document(self, url: str) -> tuple[str, str]:
        """
        Download document from URL and return file path and filename
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download document: HTTP {response.status}")
                    
                    # Extract filename from URL or generate one
                    parsed_url = urlparse(url)
                    filename = os.path.basename(parsed_url.path)
                    if not filename or '.' not in filename:
                        filename = f"document_{uuid.uuid4().hex[:8]}.pdf"
                    
                    # Create temporary file
                    temp_dir = tempfile.gettempdir()
                    file_path = os.path.join(temp_dir, f"download_{uuid.uuid4().hex}_{filename}")
                    
                    # Save the content
                    content = await response.read()
                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(content)
                    
                    return file_path, filename
                    
        except Exception as e:
            print(f"Error downloading document: {e}")
            raise Exception(f"Failed to download document from URL: {str(e)}")
    
    async def _create_mock_upload_file(self, file_path: str, filename: str):
        """
        Create a mock UploadFile object for processing
        """
        class MockUploadFile:
            def __init__(self, file_path: str, filename: str):
                self.file_path = file_path
                self.filename = filename
                self.size = os.path.getsize(file_path)
            
            async def read(self):
                async with aiofiles.open(self.file_path, 'rb') as f:
                    return await f.read()
            
            async def seek(self, position: int):
                pass
            
            async def close(self):
                pass
        
        return MockUploadFile(file_path, filename)
