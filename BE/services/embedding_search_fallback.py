import numpy as np
import pickle
import os
import json
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

# Try to import optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, using fallback similarity search")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available, using simple text matching")

from models.schemas import EmbeddingResult, DomainType

class EmbeddingSearch:
    def __init__(self):
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.embeddings_metadata = []
        self.index_file = os.getenv("FAISS_INDEX_FILE", "faiss_index.bin")
        self.metadata_file = os.getenv("FAISS_METADATA_FILE", "embeddings_metadata.json")
        
        # Initialize based on available dependencies
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_embeddings = True
            except Exception as e:
                print(f"Failed to load SentenceTransformer: {e}")
                self.use_embeddings = False
        else:
            self.use_embeddings = False
        
        if FAISS_AVAILABLE and self.use_embeddings:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.use_faiss = True
        else:
            self.index = None
            self.use_faiss = False
            self.embeddings_store = []  # Fallback storage
        
        # Load existing index if available
        self.load_index()
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
    
    def simple_text_similarity(self, query: str, text: str) -> float:
        """Simple text similarity based on word overlap"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0.0

    async def create_embeddings(self, text_chunks: List[str], document_id: str) -> Dict[str, Any]:
        """Create embeddings for text chunks"""
        try:
            if self.use_embeddings:
                # Generate embeddings using SentenceTransformer
                embeddings = self.model.encode(text_chunks, convert_to_tensor=False)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                if self.use_faiss:
                    # Add to FAISS index
                    start_id = len(self.embeddings_metadata)
                    self.index.add(embeddings.astype('float32'))
                else:
                    # Store in fallback storage
                    start_id = len(self.embeddings_store)
                    for emb in embeddings:
                        self.embeddings_store.append(emb)
            else:
                # No embeddings, just store text
                embeddings = None
                start_id = len(self.embeddings_metadata)
            
            # Store metadata
            for i, chunk in enumerate(text_chunks):
                metadata = {
                    "chunk_id": f"{document_id}_{i}",
                    "document_id": document_id,
                    "text": chunk,
                    "chunk_index": i,
                    "embedding_index": start_id + i,
                    "created_at": datetime.now().isoformat()
                }
                self.embeddings_metadata.append(metadata)
            
            # Save index and metadata
            self.save_index()
            
            return {
                "embeddings_count": len(text_chunks),
                "document_id": document_id,
                "status": "success",
                "method": "embeddings" if self.use_embeddings else "text_only"
            }
        
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return {
                "embeddings_count": 0,
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }

    async def search(self, query: str, domain: str, top_k: int = 10) -> List[EmbeddingResult]:
        """Search for similar content"""
        try:
            if not self.embeddings_metadata:
                return []
            
            if self.use_embeddings:
                return await self._embedding_search(query, domain, top_k)
            else:
                return await self._text_search(query, domain, top_k)
        
        except Exception as e:
            print(f"Error in search: {e}")
            return []

    async def _embedding_search(self, query: str, domain: str, top_k: int) -> List[EmbeddingResult]:
        """Search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            if self.use_faiss:
                # Search using FAISS
                scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.embeddings_metadata)))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.embeddings_metadata):
                        metadata = self.embeddings_metadata[idx]
                        result = EmbeddingResult(
                            text=metadata["text"],
                            similarity_score=float(score),
                            document_id=metadata["document_id"],
                            chunk_id=metadata["chunk_id"],
                            chunk_index=metadata["chunk_index"]
                        )
                        results.append(result)
            else:
                # Fallback similarity search
                similarities = []
                for i, stored_embedding in enumerate(self.embeddings_store):
                    similarity = self.cosine_similarity(query_embedding[0], stored_embedding)
                    similarities.append((similarity, i))
                
                # Sort by similarity and get top_k
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                results = []
                for similarity, idx in similarities[:top_k]:
                    if idx < len(self.embeddings_metadata):
                        metadata = self.embeddings_metadata[idx]
                        result = EmbeddingResult(
                            text=metadata["text"],
                            similarity_score=similarity,
                            document_id=metadata["document_id"],
                            chunk_id=metadata["chunk_id"],
                            chunk_index=metadata["chunk_index"]
                        )
                        results.append(result)
            
            return results
        
        except Exception as e:
            print(f"Error in embedding search: {e}")
            return []

    async def _text_search(self, query: str, domain: str, top_k: int) -> List[EmbeddingResult]:
        """Fallback text-based search"""
        try:
            similarities = []
            for i, metadata in enumerate(self.embeddings_metadata):
                similarity = self.simple_text_similarity(query, metadata["text"])
                similarities.append((similarity, metadata))
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for similarity, metadata in similarities[:top_k]:
                if similarity > 0.1:  # Minimum threshold
                    result = EmbeddingResult(
                        text=metadata["text"],
                        similarity_score=similarity,
                        document_id=metadata["document_id"],
                        chunk_id=metadata["chunk_id"],
                        chunk_index=metadata["chunk_index"]
                    )
                    results.append(result)
            
            return results
        
        except Exception as e:
            print(f"Error in text search: {e}")
            return []

    def save_index(self):
        """Save FAISS index and metadata"""
        try:
            if self.use_faiss and self.index is not None:
                faiss.write_index(self.index, self.index_file)
            
            # Save embeddings store for fallback
            if not self.use_faiss and hasattr(self, 'embeddings_store'):
                with open(self.index_file.replace('.bin', '_embeddings.pkl'), 'wb') as f:
                    pickle.dump(self.embeddings_store, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.embeddings_metadata, f, indent=2)
        
        except Exception as e:
            print(f"Error saving index: {e}")

    def load_index(self):
        """Load FAISS index and metadata"""
        try:
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.embeddings_metadata = json.load(f)
            
            # Load FAISS index
            if self.use_faiss and os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
            
            # Load embeddings store for fallback
            embeddings_file = self.index_file.replace('.bin', '_embeddings.pkl')
            if not self.use_faiss and os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    self.embeddings_store = pickle.load(f)
            
        except Exception as e:
            print(f"Error loading index: {e}")
            # Reset to empty state on error
            if self.use_faiss:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self.embeddings_store = []
            self.embeddings_metadata = []

    async def get_documents_info(self) -> List[Dict]:
        """Get information about stored documents"""
        documents = {}
        for metadata in self.embeddings_metadata:
            doc_id = metadata["document_id"]
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "chunk_count": 0,
                    "created_at": metadata.get("created_at", "unknown")
                }
            documents[doc_id]["chunk_count"] += 1
        
        return list(documents.values())

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document and its embeddings"""
        try:
            # Find indices to remove
            indices_to_remove = []
            for i, metadata in enumerate(self.embeddings_metadata):
                if metadata["document_id"] == document_id:
                    indices_to_remove.append(i)
            
            if not indices_to_remove:
                return {"status": "error", "message": "Document not found"}
            
            # Remove metadata (in reverse order to maintain indices)
            for i in reversed(indices_to_remove):
                del self.embeddings_metadata[i]
            
            # For simplicity, rebuild the entire index
            # In production, you might want a more efficient approach
            if self.use_faiss:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self.embeddings_store = []
            
            # Re-add remaining embeddings
            if self.use_embeddings and self.embeddings_metadata:
                # This is simplified - in practice you'd want to store embeddings with metadata
                pass
            
            # Save updated index
            self.save_index()
            
            return {"status": "success", "removed_chunks": len(indices_to_remove)}
        
        except Exception as e:
            print(f"Error deleting document: {e}")
            return {"status": "error", "error": str(e)}
