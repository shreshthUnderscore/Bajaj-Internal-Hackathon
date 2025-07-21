import logging
import os
import json
from typing import List, Dict, Any

# Import the Google Generative AI library
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMDecisionMaker:
    """
    A class to interact with a Large Language Model (LLM) for contextual decision making
    and rationale generation based on retrieved document chunks.
    """
    def __init__(self):
        """
        Initializes the LLMDecisionMaker.
        Retrieves the Gemini API key from environment variables.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logging.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
            raise ValueError("GEMINI_API_KEY is not set.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash') # Using gemini-2.0-flash as specified

        logging.info("LLMDecisionMaker initialized with Gemini API.")

    def _construct_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Constructs the prompt for the LLM.
        The prompt includes the user's query and the content of the retrieved chunks.
        """
        context_str = "\n\n--- Retrieved Document Chunks ---\n\n"
        if not retrieved_chunks:
            context_str += "No relevant documents found. The answer will be based on general knowledge.\n"
        else:
            for i, chunk in enumerate(retrieved_chunks):
                doc_source = chunk.get('source', 'Unknown Document')
                chunk_id = chunk.get('chunk_id', f'Chunk {i+1}')
                content = chunk.get('full_chunk_content', 'No content available.')
                context_str += f"Document: {doc_source}, Chunk ID: {chunk_id}\nContent:\n{content}\n\n"
            context_str += "---------------------------------\n\n"

        prompt = f"""
You are an intelligent query retrieval system designed for insurance, legal, HR, and compliance domains.
Your task is to answer user queries based *only* on the provided retrieved document chunks.
If the information is not explicitly available in the provided chunks, state that you cannot find the answer in the given context.

After providing the answer, you MUST provide a detailed rationale explaining *why* you arrived at that answer.
The rationale should clearly reference the specific document(s) and relevant sentences/clauses from the provided chunks that support your answer.
If you combine information from multiple chunks, explain how they connect.

Your output MUST be a JSON object with the following structure:
{{
  "query": "original user query",
  "answer": "your concise answer based on context",
  "rationale": "detailed explanation of your decision, referencing specific document parts",
  "relevant_chunks_used": [
    {{
      "chunk_id": "ID of the chunk used",
      "text_snippet": "exact sentence/paragraph from the chunk that supports the answer"
    }}
    // ... potentially more chunks
  ]
}}

Here is the user's query: "{query}"

{context_str}

Please provide your response in the specified JSON format.
"""
        logging.debug(f"Constructed LLM prompt:\n{prompt}")
        return prompt

    async def get_decision_and_rationale(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calls the LLM to get a contextual decision and rationale.
        Args:
            query (str): The user's natural language query.
            retrieved_chunks (List[Dict[str, Any]]): A list of dictionaries representing
                                                      the semantically retrieved document chunks.
        Returns:
            Dict[str, Any]: A dictionary containing the LLM's answer, rationale,
                            and referenced chunks, or an error message.
        """
        prompt = self._construct_prompt(query, retrieved_chunks)
        
        try:
            # DIAGNOSTIC CHANGE: Removed 'await' to see if generate_content is synchronous
            response = self.model.generate_content(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING"},
                            "answer": {"type": "STRING"},
                            "rationale": {"type": "STRING"},
                            "relevant_chunks_used": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "chunk_id": {"type": "STRING"},
                                        "text_snippet": {"type": "STRING"}
                                    }
                                }
                            }
                        }
                    }
                }
            )
            
            # The response.text property will contain the JSON string
            llm_response_text = response.text
            
            # Parse the JSON string from the LLM
            try:
                parsed_json_response = json.loads(llm_response_text)
                logging.info("Successfully received and parsed LLM response.")
                return parsed_json_response
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response as JSON: {e}. Raw response: {llm_response_text}")
                return {"error": "Failed to parse LLM response as JSON.", "raw_response": llm_response_text}

        except Exception as e:
            logging.error(f"Error calling Gemini API: {e}")
            return {"error": f"Error calling Gemini API: {e}"}


if __name__ == "__main__":
    import asyncio
    # For testing, we'll need to simulate retrieved chunks
    # and use the SemanticSearcher to get actual chunks.
    from semantic_search import SemanticSearcher
    from document_processor import process_document
    # from document_loader import load_document # Not needed for this test block's dummy data

    print("--- Testing LLM Contextual Decision Maker Module ---")

    # Setup dummy data, similar to semantic_search.py's test block
    EMBEDDING_DIMENSION = 384
    FAISS_INDEX_FILE = "hackathon_faiss_index.bin"

    # Clean up previous test index files if they exist
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)
    if os.path.exists(FAISS_INDEX_FILE.replace(".bin", "_metadata.json")):
        os.remove(FAISS_INDEX_FILE.replace(".bin", "_metadata.json"))

    # Initialize SemanticSearcher (which initializes EmbeddingGenerator and FAISSVectorDBManager internally)
    print("\nInitializing SemanticSearcher and LLMDecisionMaker...")
    try:
        searcher = SemanticSearcher(
            embedding_model_name='all-MiniLM-L6-v2',
            faiss_index_path=FAISS_INDEX_FILE
        )
        llm_decision_maker = LLMDecisionMaker()
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        exit()

    # Prepare sample documents (using dummy data for demonstration)
    print("\nPreparing and processing sample documents for indexing...")
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

    # Generate embeddings for all chunks and add to FAISS using the searcher's internal components
    print(f"\nGenerating embeddings for {len(all_processed_chunks)} chunks and adding to FAISS...")
    chunk_texts = [chunk['content'] for chunk in all_processed_chunks]
    chunk_metadatas = [chunk['metadata'] for chunk in all_processed_chunks]
    
    chunk_embeddings = searcher.embedding_generator.generate_embeddings(chunk_texts)
    
    for i, chunk_meta in enumerate(chunk_metadatas):
        chunk_meta['full_chunk_content'] = chunk_texts[i] # Add full content to metadata for LLM context

    searcher.faiss_db_manager.add_vectors(chunk_embeddings, chunk_metadatas)
    print(f"Total vectors in FAISS index: {searcher.faiss_db_manager.get_total_vectors()}")

    # Perform a sample query and get LLM decision
    async def run_test_query():
        test_query = "What is the deductible for accidental damage in the insurance policy?"
        print(f"\nPerforming semantic search for query: '{test_query}'")
        retrieved_chunks = searcher.search(test_query, k=2)

        if retrieved_chunks:
            print(f"Retrieved {len(retrieved_chunks)} chunks. Passing to LLM...")
            llm_response = await llm_decision_maker.get_decision_and_rationale(test_query, retrieved_chunks)
            print("\n--- LLM Response ---")
            print(json.dumps(llm_response, indent=2))
        else:
            print("No chunks retrieved for the query. Cannot ask LLM.")
            llm_response = await llm_decision_maker.get_decision_and_rationale(test_query, [])
            print("\n--- LLM Response (No context) ---")
            print(json.dumps(llm_response, indent=2))


    # Run the asynchronous test function
    asyncio.run(run_test_query())

    # Clean up test files
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)
    if os.path.exists(FAISS_INDEX_FILE.replace(".bin", "_metadata.json")):
        os.remove(FAISS_INDEX_FILE.replace(".bin", "_metadata.json"))
