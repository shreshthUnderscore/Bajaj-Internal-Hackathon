import numpy as np
import faiss
import pickle
import os
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import asyncio
from datetime import datetime

from models.schemas import EmbeddingResult, DomainType

class EmbeddingSearch:
    def __init__(self):
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Storage for metadata
        self.embeddings_metadata = []
        self.index_file = "faiss_index.bin"
        self.metadata_file = "embeddings_metadata.json"
        
        # Load existing index if available
        self.load_index()
    
    async def create_embeddings(self, text_chunks: List[str], document_id: str) -> Dict[str, Any]:
        """
        Create embeddings for text chunks and add to FAISS index
        """
        try:
            # Generate embeddings
            embeddings = self.model.encode(text_chunks, convert_to_tensor=False)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to FAISS index
            start_id = len(self.embeddings_metadata)
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata
            for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
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
                "embeddings_count": len(embeddings),
                "document_id": document_id,
                "status": "success"
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
        """
        Search for similar embeddings using FAISS
        """
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Prepare results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.embeddings_metadata):
                    metadata = self.embeddings_metadata[idx]
                    result = EmbeddingResult(
                        chunk_id=metadata["chunk_id"],
                        text=metadata["text"],
                        similarity_score=float(similarity),
                        document_id=metadata["document_id"],
                        metadata={
                            "chunk_index": metadata["chunk_index"],
                            "created_at": metadata["created_at"]
                        }
                    )
                    results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return results
        
        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return []
    
    async def search_by_document(self, document_id: str, query: str, top_k: int = 5) -> List[EmbeddingResult]:
        """
        Search within a specific document
        """
        # Get all results first
        all_results = await self.search(query, "", top_k * 3)  # Get more to filter by document
        
        # Filter by document_id
        document_results = [r for r in all_results if r.document_id == document_id]
        
        return document_results[:top_k]
    
    async def delete_embeddings(self, document_id: str) -> Dict[str, Any]:
        """
        Delete embeddings for a specific document
        Note: FAISS doesn't support deletion, so we rebuild the index
        """
        try:
            # Filter out embeddings for the specified document
            filtered_metadata = [
                meta for meta in self.embeddings_metadata 
                if meta["document_id"] != document_id
            ]
            
            if len(filtered_metadata) == len(self.embeddings_metadata):
                return {"status": "not_found", "document_id": document_id}
            
            # Rebuild index if there are remaining embeddings
            if filtered_metadata:
                # Create new index
                new_index = faiss.IndexFlatIP(self.embedding_dim)
                new_metadata = []
                
                # Re-encode and add remaining chunks
                texts_to_reindex = [meta["text"] for meta in filtered_metadata]
                if texts_to_reindex:
                    embeddings = self.model.encode(texts_to_reindex, convert_to_tensor=False)
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    new_index.add(embeddings.astype('float32'))
                    
                    # Update metadata with new indices
                    for i, meta in enumerate(filtered_metadata):
                        meta["embedding_index"] = i
                        new_metadata.append(meta)
                
                self.index = new_index
                self.embeddings_metadata = new_metadata
            else:
                # No embeddings left, create empty index
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.embeddings_metadata = []
            
            # Save updated index
            self.save_index()
            
            return {
                "status": "success", 
                "document_id": document_id,
                "remaining_embeddings": len(self.embeddings_metadata)
            }
        
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
            return {"status": "error", "error": str(e)}
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.embeddings_metadata, f, indent=2)
        
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
            
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.embeddings_metadata = json.load(f)
        
        except Exception as e:
            print(f"Error loading index: {e}")
            # Create new index if loading fails
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.embeddings_metadata = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding index"""
        document_counts = {}
        for meta in self.embeddings_metadata:
            doc_id = meta["document_id"]
            document_counts[doc_id] = document_counts.get(doc_id, 0) + 1
        
        return {
            "total_embeddings": len(self.embeddings_metadata),
            "total_documents": len(document_counts),
            "index_size": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "document_breakdown": document_counts
        }
    
    async def semantic_search_with_filters(
        self, 
        query: str, 
        domain: str, 
        document_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[EmbeddingResult]:
        """
        Search with additional filters
        """
        # Get initial results
        results = await self.search(query, domain, top_k * 2)  # Get more to apply filters
        
        # Apply document filter if specified
        if document_ids:
            results = [r for r in results if r.document_id in document_ids]
        
        return results[:top_k]
