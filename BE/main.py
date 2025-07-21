import os
import logging
import asyncio
import tempfile
import httpx # For asynchronous HTTP requests
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import our core modules
from document_loader import load_document
from document_processor import process_document
from semantic_search import SemanticSearcher
from llm_decision_maker import LLMDecisionMaker

# Load environment variables (including GEMINI_API_KEY and AUTH_TOKEN)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Intelligent Query Retrieval System",
    description="API for processing large documents and answering contextual queries.",
    version="1.0.0"
)

# --- Global/Singleton Instances of Core Components ---
# These will be initialized once when the application starts
semantic_searcher: Optional[SemanticSearcher] = None
llm_decision_maker: Optional[LLMDecisionMaker] = None

# Define FAISS index file path
FAISS_INDEX_FILE = "api_faiss_index.bin"
EMBEDDING_DIMENSION = 384 # Matches 'all-MiniLM-L6-v2'

@app.on_event("startup")
async def startup_event():
    """
    Initializes the SemanticSearcher and LLMDecisionMaker when the FastAPI app starts.
    This ensures models are loaded and FAISS index is ready.
    """
    global semantic_searcher, llm_decision_maker

    logging.info("Initializing core components...")
    try:
        # Clean up existing FAISS index files on startup for a fresh start in hackathon context
        # In production, you might manage index persistence differently.
        if os.path.exists(FAISS_INDEX_FILE):
            os.remove(FAISS_INDEX_FILE)
        if os.path.exists(FAISS_INDEX_FILE.replace(".bin", "_metadata.json")):
            os.remove(FAISS_INDEX_FILE.replace(".bin", "_metadata.json"))
        
        semantic_searcher = SemanticSearcher(
            embedding_model_name='all-MiniLM-L6-v2',
            faiss_index_path=FAISS_INDEX_FILE
        )
        llm_decision_maker = LLMDecisionMaker()
        logging.info("Core components initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize core components on startup: {e}")
        # Depending on severity, you might want to raise an exception to prevent app from starting
        raise HTTPException(status_code=500, detail=f"Server startup failed: {e}")


# --- Authentication Dependency ---
# For hackathon, a simple bearer token check from environment variable
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "52d676a251b57c9a1ac3603f7a67b3c960fba5de52faccd7267abcc49f9fcc50")

async def verify_token(authorization: str = Header(...)):
    """
    Dependency to verify the Authorization Bearer token.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or malformed.")
    
    token = authorization.split(" ")[1]
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authentication token.")
    return True

# --- Pydantic Models for Request and Response ---
class RunRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)]) # Updated path
async def run_submissions(request: RunRequest):
    """
    Processes a document from a URL and answers a list of questions using LLM-powered retrieval.
    """
    if not semantic_searcher or not llm_decision_maker:
        raise HTTPException(status_code=500, detail="Core components not initialized. Server startup failed.")

    document_url = request.documents
    questions = request.questions
    all_answers = []

    logging.info(f"Received request for document: {document_url} with {len(questions)} questions.")

    # 1. Download the document
    temp_pdf_path = None
    try:
        async with httpx.AsyncClient() as client:
            logging.info(f"Downloading document from {document_url}...")
            response = await client.get(document_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            # Save the downloaded content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_pdf_path = temp_file.name
            logging.info(f"Document downloaded to temporary file: {temp_pdf_path}")

        # 2. Load, Process, and Index the document
        logging.info(f"Loading and processing document: {temp_pdf_path}")
        loaded_doc = load_document(temp_pdf_path)
        if not loaded_doc["text_content"]:
            raise HTTPException(status_code=400, detail=f"Could not extract text from document: {document_url}. It might be a scanned PDF requiring OCR, or the file is corrupted/empty.")

        processed_chunks = process_document(loaded_doc)
        if not processed_chunks:
            raise HTTPException(status_code=400, detail=f"No processable chunks found in document: {document_url}")

        # Extract texts and metadatas for embedding and indexing
        chunk_texts = [chunk['content'] for chunk in processed_chunks]
        chunk_metadatas = [chunk['metadata'] for chunk in processed_chunks]
        
        # Generate embeddings
        chunk_embeddings = semantic_searcher.embedding_generator.generate_embeddings(chunk_texts)
        if not chunk_embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings for document chunks.")

        # Add full content to metadata for LLM context
        for i, chunk_meta in enumerate(chunk_metadatas):
            chunk_meta['full_chunk_content'] = chunk_texts[i]

        # Clear existing index and add new vectors for this document
        # In a multi-document or persistent scenario, you'd append or manage multiple indexes.
        # For hackathon, assuming one document at a time or re-indexing per request.
        semantic_searcher.faiss_db_manager.clear_index() # Clear previous data
        semantic_searcher.faiss_db_manager.add_vectors(chunk_embeddings, chunk_metadatas)
        logging.info(f"Document indexed. Total vectors in FAISS: {semantic_searcher.faiss_db_manager.get_total_vectors()}")

        # 3. Process each question
        for i, question in enumerate(questions):
            logging.info(f"Processing question {i+1}/{len(questions)}: '{question}'")
            # Perform semantic search to get relevant chunks
            retrieved_chunks = semantic_searcher.search(question, k=3) # Retrieve top 3 chunks

            # Get LLM decision and rationale
            llm_response = await llm_decision_maker.get_decision_and_rationale(question, retrieved_chunks)
            
            # Extract the answer
            answer = llm_response.get("answer", "Could not find a specific answer in the provided context.")
            all_answers.append(answer)
            logging.info(f"Answer for '{question}': {answer[:100]}...") # Log first 100 chars

    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error downloading document: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error downloading document: {e.response.text}")
    except httpx.RequestError as e:
        logging.error(f"Network error downloading document: {e}")
        raise HTTPException(status_code=500, detail=f"Network error downloading document: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        # Clean up the temporary PDF file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logging.info(f"Cleaned up temporary file: {temp_pdf_path}")

    return RunResponse(answers=all_answers)

# --- Root Endpoint for Health Check ---
@app.get("/")
async def read_root():
    return {"message": "LLM-Powered Intelligent Query Retrieval System is running!"}

httpx==0.27.0

