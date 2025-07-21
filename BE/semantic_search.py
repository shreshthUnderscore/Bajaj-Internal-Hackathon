import logging
import os
from typing import List, Dict, Any, Tuple

# Import classes from our previously created modules
from embedding_generator import EmbeddingGenerator
from vector_db_manager import FAISSVectorDBManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticSearcher:
    """
    Orchestrates semantic search by generating query embeddings and querying the FAISS index.
    """
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2',
                 faiss_index_path: str = "faiss_index.bin"):
        """
        Initializes the SemanticSearcher with an EmbeddingGenerator and FAISSVectorDBManager.
        Args:
            embedding_model_name (str): The name of the Sentence-Transformer model to use.
            faiss_index_path (str): The file path for the FAISS index.
        """
        try:
            self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
            # Ensure the FAISS manager is initialized with the correct embedding dimension
            self.faiss_db_manager = FAISSVectorDBManager(
                embedding_dimension=self.embedding_generator.embedding_dimension,
                index_path=faiss_index_path
            )
            logging.info("SemanticSearcher initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize SemanticSearcher: {e}")
            raise

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search for relevant document chunks based on a natural language query.
        Args:
            query (str): The natural language query.
            k (int): The number of top relevant chunks to retrieve.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
                                  the metadata of a retrieved chunk, including its content.
                                  The list is sorted by relevance (distance).
        """
        if not query:
            logging.warning("Empty query provided for semantic search.")
            return []

        try:
            # 1. Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embeddings([query])
            if not query_embedding:
                logging.error("Failed to generate embedding for the query.")
                return []
            
            # generate_embeddings returns a list of embeddings, we need the first one
            query_embedding = query_embedding[0] 

            # 2. Search the FAISS index
            # The search_vectors method returns (distance, metadata_dict)
            raw_results: List[Tuple[float, Dict[str, Any]]] = self.faiss_db_manager.search_vectors(query_embedding, k=k)

            # 3. Format results for output
            formatted_results = []
            for distance, metadata in raw_results:
                # The metadata dictionary already contains 'content' and other relevant info
                # from the document_processor module.
                metadata['distance'] = float(distance) # Ensure distance is a standard float
                formatted_results.append(metadata)
            
            # Sort by distance (lower distance means higher similarity in L2)
            formatted_results.sort(key=lambda x: x['distance'])

            logging.info(f"Semantic search completed for query: '{query}'. Retrieved {len(formatted_results)} results.")
            return formatted_results

        except Exception as e:
            logging.error(f"Error during semantic search for query '{query}': {e}")
            return []

if __name__ == "__main__":
    # This block demonstrates the full ingestion and search pipeline.
    print("--- End-to-End Semantic Search Test ---")

    # Ensure necessary modules are available for import
    # from document_loader import load_document # No longer directly loading files in this test block
    from document_processor import process_document
    # EmbeddingGenerator and FAISSVectorDBManager are imported at the top

    EMBEDDING_DIMENSION = 384 # Matches 'all-MiniLM-L6-v2'
    FAISS_INDEX_FILE = "hackathon_faiss_index.bin"

    # Clean up previous test index files if they exist
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)
    if os.path.exists(FAISS_INDEX_FILE.replace(".bin", "_metadata.json")):
        os.remove(FAISS_INDEX_FILE.replace(".bin", "_metadata.json"))

    # 1. Initialize components
    print("\nInitializing SemanticSearcher...")
    try:
        # Initialize the searcher directly, it will create its own embedding_generator and faiss_db_manager
        searcher = SemanticSearcher(
            embedding_model_name='all-MiniLM-L6-v2',
            faiss_index_path=FAISS_INDEX_FILE
        )
        # We no longer need separate embedding_gen and faiss_db_manager here
        # embedding_gen = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
        # faiss_db_manager = FAISSVectorDBManager(EMBEDDING_DIMENSION, FAISS_INDEX_FILE)
    except Exception as e:
        print(f"Failed to initialize core components: {e}")
        exit()

    # 2. Prepare sample documents (using dummy data for demonstration)
    print("\nPreparing and processing sample documents...")
    sample_docs_raw = [
        {
            "file_path": "doc_insurance_policy.txt",
            "document_type": "txt",
            "text_content": """
            This is an insurance policy document.
            Section 3.1: Accidental Damage Coverage. This policy covers accidental damage to your vehicle.
            A deductible of $500 applies to all accidental damage claims.
            Claims must be filed within 30 days of the incident.
            Section 3.2: Exclusions. Damage due to intentional acts or natural disasters like floods is excluded.
            Policy Number: INS-XYZ-2025. Effective Date: 2025-01-01.
            """
        },
        {
            "file_path": "doc_hr_handbook.txt",
            "document_type": "txt",
            "text_content": """
            Company Employee Handbook.
            Chapter 5: Leave Policy. Employees are entitled to 10 days of paid sick leave per year.
            Parental leave for fathers is 8 weeks paid. Maternity leave is 16 weeks paid.
            All leave requests must be submitted through the HR portal.
            This handbook is compliant with local labor laws.
            """
        },
        {
            "file_path": "doc_legal_contract.txt",
            "document_type": "txt",
            "text_content": """
            Sales Agreement Contract.
            Clause 7.1: Force Majeure. Neither party shall be liable for any failure to perform
            its obligations where such failure is caused by circumstances beyond its reasonable control,
            including but not limited to acts of God, war, or unforeseeable supply chain disruptions.
            Clause 8.2: Dispute Resolution. Any disputes arising under this contract shall be settled by arbitration.
            """
        },
        {
            "file_path": "doc_compliance_gdpr.txt",
            "document_type": "txt",
            "text_content": """
            Internal Data Privacy Compliance Document.
            This document outlines our adherence to GDPR Article 17, the 'right to erasure'.
            Data subjects can request deletion of their personal data.
            Such requests will be processed within 30 days, unless legal obligations prevent deletion.
            We also comply with ISO 27001 standards.
            """
        }
    ]

    all_processed_chunks = []
    for doc_data in sample_docs_raw:
        simulated_loaded_doc = {
            "file_path": doc_data["file_path"],
            "document_type": doc_data["document_type"],
            "text_content": doc_data["text_content"],
            "document_id": os.path.basename(doc_data["file_path"])
        }
        
        processed_chunks = process_document(simulated_loaded_doc)
        all_processed_chunks.extend(processed_chunks)

    # 3. Generate embeddings for all chunks and add to FAISS using the searcher's internal components
    print(f"\nGenerating embeddings for {len(all_processed_chunks)} chunks and adding to FAISS...")
    chunk_texts = [chunk['content'] for chunk in all_processed_chunks]
    chunk_metadatas = [chunk['metadata'] for chunk in all_processed_chunks]
    
    # Use the searcher's embedding_generator
    chunk_embeddings = searcher.embedding_generator.generate_embeddings(chunk_texts)
    
    for i, chunk_meta in enumerate(chunk_metadatas):
        chunk_meta['full_chunk_content'] = chunk_texts[i]

    # Use the searcher's faiss_db_manager to add vectors
    searcher.faiss_db_manager.add_vectors(chunk_embeddings, chunk_metadatas)
    print(f"Total vectors in FAISS index: {searcher.faiss_db_manager.get_total_vectors()}")

    # 4. Perform semantic searches
    print("\n--- Performing Semantic Searches ---")

    queries = [
        "What is the deductible for car damage due to an accident?", # Insurance
        "How much paid leave do fathers get?", # HR
        "What happens if there's a supply chain problem in the contract?", # Legal
        "Tell me about the right to be forgotten under data regulations.", # Compliance
        "What is covered by the policy?", # General insurance
        "How are disputes resolved in the agreement?" # General legal
    ]

    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: '{query}'")
        results = searcher.search(query, k=2) # Retrieve top 2 results

        if results:
            for j, res in enumerate(results):
                print(f"  Result {j+1}:")
                print(f"    Distance: {res['distance']:.4f}")
                print(f"    Document: {res.get('source', 'N/A')}")
                print(f"    Chunk ID: {res.get('chunk_id', 'N/A')}")
                print(f"    Content: {res.get('full_chunk_content', 'N/A')[:200]}...")
                print(f"    Domain: {res.get('domain', 'N/A')}")
        else:
            print("  No relevant results found.")

    # Clean up test files
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)
    if os.path.exists(FAISS_INDEX_FILE.replace(".bin", "_metadata.json")):
        os.remove(FAISS_INDEX_FILE.replace(".bin", "_metadata.json"))
