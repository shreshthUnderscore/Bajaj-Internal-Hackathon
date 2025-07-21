import faiss
import numpy as np
import logging
import os
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FAISSVectorDBManager:
    """
    Manages a FAISS index for storing and retrieving vector embeddings.
    """
    def __init__(self, embedding_dimension: int, index_path: str = "faiss_index.bin"):
        """
        Initializes the FAISS index.
        Args:
            embedding_dimension (int): The dimension of the embeddings to be stored.
                                       This must match the output dimension of your embedding model.
            index_path (str): The file path to save/load the FAISS index.
        """
        self.embedding_dimension = embedding_dimension
        self.index_path = index_path
        self.index = None
        self.metadata_store: List[Dict[str, Any]] = [] # To store metadata corresponding to each vector

        self._load_or_create_index()

    def _load_or_create_index(self):
        """
        Loads the FAISS index from disk if it exists, otherwise creates a new one.
        """
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                # Load metadata. We'll store metadata in a separate file or a simple list for now.
                # In a real system, you might use a proper database for metadata.
                metadata_path = self.index_path.replace(".bin", "_metadata.json")
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata_store = json.load(f)
                    logging.info(f"Loaded FAISS index and metadata from {self.index_path} and {metadata_path}.")
                else:
                    logging.warning(f"Metadata file not found at {metadata_path}. Index loaded, but metadata is empty.")
                    self.metadata_store = []
            except Exception as e:
                logging.error(f"Error loading FAISS index from {self.index_path}: {e}. Creating a new index.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """
        Creates a new FAISS index.
        We'll use IndexFlatL2 for simplicity, which performs exhaustive L2 (Euclidean) distance search.
        For larger datasets, consider IndexIVFFlat or IndexHNSW for faster approximate nearest neighbor search.
        """
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.metadata_store = []
        logging.info(f"Created a new FAISS index with dimension {self.embedding_dimension}.")

    def add_vectors(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Adds vectors (embeddings) and their corresponding metadata to the FAISS index.
        Args:
            embeddings (List[List[float]]): A list of embedding vectors.
            metadatas (List[Dict[str, Any]]): A list of metadata dictionaries,
                                               one for each embedding, in the same order.
        """
        if not embeddings:
            logging.warning("No embeddings provided to add.")
            return

        if len(embeddings) != len(metadatas):
            logging.error("Number of embeddings and metadatas must match.")
            return

        # Convert list of lists to numpy array
        vectors = np.array(embeddings).astype('float32')

        if vectors.shape[1] != self.embedding_dimension:
            logging.error(f"Embedding dimension mismatch. Expected {self.embedding_dimension}, got {vectors.shape[1]}.")
            return

        try:
            self.index.add(vectors)
            self.metadata_store.extend(metadatas)
            self._save_index() # Save after adding
            logging.info(f"Added {len(embeddings)} vectors to the FAISS index. Total vectors: {self.index.ntotal}")
        except Exception as e:
            logging.error(f"Error adding vectors to FAISS index: {e}")

    def search_vectors(self, query_embedding: List[float], k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Searches the FAISS index for the top-k nearest neighbors to a query embedding.
        Args:
            query_embedding (List[float]): The embedding of the query.
            k (int): The number of nearest neighbors to retrieve.
        Returns:
            List[Tuple[float, Dict[str, Any]]]: A list of tuples, where each tuple contains
                                                 (distance, metadata_of_retrieved_chunk).
        """
        if not self.index or self.index.ntotal == 0:
            logging.warning("FAISS index is empty or not initialized. Cannot perform search.")
            return []

        query_vector = np.array([query_embedding]).astype('float32')
        if query_vector.shape[1] != self.embedding_dimension:
            logging.error(f"Query embedding dimension mismatch. Expected {self.embedding_dimension}, got {query_vector.shape[1]}.")
            return []

        try:
            distances, indices = self.index.search(query_vector, k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1: # -1 indicates no valid result
                    # Ensure the index is within bounds of metadata_store
                    if 0 <= idx < len(self.metadata_store):
                        results.append((distances[0][i], self.metadata_store[idx]))
                    else:
                        logging.warning(f"Retrieved index {idx} out of bounds for metadata store (size {len(self.metadata_store)}).")
            logging.info(f"Searched FAISS index for top {k} results.")
            return results
        except Exception as e:
            logging.error(f"Error searching FAISS index: {e}")
            return []

    def _save_index(self):
        """
        Saves the FAISS index and its associated metadata to disk.
        """
        try:
            faiss.write_index(self.index, self.index_path)
            metadata_path = self.index_path.replace(".bin", "_metadata.json")
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_store, f, indent=4)
            logging.info(f"FAISS index and metadata saved to {self.index_path} and {metadata_path}.")
        except Exception as e:
            logging.error(f"Error saving FAISS index: {e}")

    def get_total_vectors(self) -> int:
        """Returns the total number of vectors in the index."""
        return self.index.ntotal if self.index else 0

    def clear_index(self):
        """Clears the FAISS index and metadata."""
        self._create_new_index() # Re-initialize to an empty index
        self._save_index() # Save the empty index to disk
        logging.info("FAISS index and metadata cleared.")


if __name__ == "__main__":
    # Example usage:
    print("--- Testing FAISS Vector DB Manager Module ---")

    # Assuming embedding dimension from all-MiniLM-L6-v2
    EMBEDDING_DIMENSION = 384
    INDEX_FILE = "test_faiss_index.bin"

    # Clean up previous test files if they exist
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(INDEX_FILE.replace(".bin", "_metadata.json")):
        os.remove(INDEX_FILE.replace(".bin", "_metadata.json"))

    # 1. Initialize the FAISS manager
    print("\nInitializing FAISSVectorDBManager...")
    db_manager = FAISSVectorDBManager(EMBEDDING_DIMENSION, INDEX_FILE)
    print(f"Total vectors after init: {db_manager.get_total_vectors()}")

    # 2. Prepare dummy embeddings and metadata
    dummy_embeddings = [
        [0.1, 0.2, 0.3, 0.4] * (EMBEDDING_DIMENSION // 4), # Ensure correct dimension
        [0.9, 0.8, 0.7, 0.6] * (EMBEDDING_DIMENSION // 4),
        [0.2, 0.3, 0.4, 0.5] * (EMBEDDING_DIMENSION // 4),
        [0.8, 0.7, 0.6, 0.5] * (EMBEDDING_DIMENSION // 4),
    ]
    dummy_metadatas = [
        {"chunk_id": "doc1_chunk_0", "content": "This is about insurance policy terms."},
        {"chunk_id": "doc1_chunk_1", "content": "This is about legal contract clauses."},
        {"chunk_id": "doc2_chunk_0", "content": "HR policies on leave."},
        {"chunk_id": "doc2_chunk_1", "content": "Compliance regulations for data."},
    ]

    # 3. Add vectors
    print("\nAdding dummy vectors...")
    db_manager.add_vectors(dummy_embeddings, dummy_metadatas)
    print(f"Total vectors after adding: {db_manager.get_total_vectors()}")

    # 4. Test loading from disk (simulate re-initialization)
    print("\nRe-initializing manager to test loading from disk...")
    new_db_manager = FAISSVectorDBManager(EMBEDDING_DIMENSION, INDEX_FILE)
    print(f"Total vectors after re-init: {new_db_manager.get_total_vectors()}")
    print(f"Metadata store size after re-init: {len(new_db_manager.metadata_store)}")
    if new_db_manager.metadata_store:
        print(f"First loaded metadata: {new_db_manager.metadata_store[0]}")

    # 5. Perform a search
    print("\nPerforming a search...")
    query_embedding = [0.15, 0.25, 0.35, 0.45] * (EMBEDDING_DIMENSION // 4) # Similar to first dummy embedding
    search_results = new_db_manager.search_vectors(query_embedding, k=2)

    print(f"\nSearch results (top {len(search_results)}):")
    for distance, metadata in search_results:
        print(f"  Distance: {distance:.4f}, Chunk ID: {metadata.get('chunk_id')}, Content: {metadata.get('content')[:50]}...")

    # 6. Test search on an empty index
    print("\nClearing index and testing search on empty index...")
    db_manager.clear_index()
    print(f"Total vectors after clearing: {db_manager.get_total_vectors()}")
    empty_search_results = db_manager.search_vectors(query_embedding, k=1)
    print(f"Search results from empty index: {empty_search_results}")

    # Clean up test files
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(INDEX_FILE.replace(".bin", "_metadata.json")):
        os.remove(INDEX_FILE.replace(".bin", "_metadata.json"))
