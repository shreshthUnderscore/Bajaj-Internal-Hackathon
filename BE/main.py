from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- LangChain and ML Components ---
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    text: str

class SourceDocument(BaseModel):
    source: str
    page_content: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    source_documents: list[SourceDocument]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="An API for processing documents and answering contextual queries.",
    version="1.0.0"
)

# --- Global Variables & Setup ---
# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store.faiss"
LLM_REPO_ID = "google/flan-t5-xxl" # A powerful, instruction-tuned model

# Check for Hugging Face API Token
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    # In a real app, you'd use a more robust configuration system
    print("WARNING: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
    print("The application will run, but queries will fail.")
    print("Get a token from https://huggingface.co/settings/tokens and set it.")

qa_chain = None
try:
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"Loading vector store from {VECTOR_STORE_PATH}...")
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTOR_STORE_PATH}'. "
            "Please run 'python BE/ingest.py' to create it first."
        )
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)

    print("Loading LLM from Hugging Face Hub...")
    llm = HuggingFaceHub(
        repo_id=LLM_REPO_ID,
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )

    # Create a RetrievalQA chain
    # This chain combines a retriever (our vector store) and an LLM.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" puts all retrieved chunks into the context
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 chunks
        return_source_documents=True # Include source documents in the result
    )
    print("API is ready to accept queries.")

except Exception as e:
    print(f"FATAL: Error during model loading: {e}")
    print("The API will start, but the /query endpoint will not work.")

# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
async def root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "API is running."}

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def handle_query(request: QueryRequest):
    """
    Receives a natural language query and returns a contextual answer.
    """
    if not qa_chain:
        raise HTTPException(
            status_code=503, 
            detail="QA chain is not available due to a startup error. Check server logs."
        )

    if not request.text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    print(f"Received query: {request.text}")

    try:
        # --- Run the QA chain ---
        result = qa_chain({"query": request.text})

        # --- Format and return the response ---
        # The 'result' key contains the LLM's answer.
        # The 'source_documents' key contains the retrieved chunks.
        return {
            "query": request.text,
            "answer": result.get("result", "No answer could be generated."),
            "source_documents": [
                {
                    "source": doc.metadata.get("source", "Unknown").replace("data/", ""),
                    "page_content": doc.page_content
                }
                for doc in result.get("source_documents", [])
            ]
        }
    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process the query: {e}")

# To run this app:
# 1. Install dependencies: pip install -r requirements.txt
# 2. Run the server: uvicorn main:app --reload
