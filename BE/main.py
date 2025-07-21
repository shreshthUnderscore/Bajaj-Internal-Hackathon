from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from services.document_processor import DocumentProcessor
from services.llm_parser import LLMParser
from services.embedding_search import EmbeddingSearch
from services.clause_matcher import ClauseMatcher
from services.logic_evaluator import LogicEvaluator
from services.url_document_processor import URLDocumentProcessor
from models.schemas import QueryRequest, QueryResponse, DocumentInfo, HackRXRequest, HackRXResponse, HackRXAnswer, SimplifiedHackRXResponse

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced document processing and query retrieval for insurance, legal, HR, and compliance domains",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
llm_parser = LLMParser()
embedding_search = EmbeddingSearch()
clause_matcher = ClauseMatcher()
logic_evaluator = LogicEvaluator()
url_document_processor = URLDocumentProcessor()

@app.get("/")
async def root():
    return {"message": "LLM-Powered Intelligent Query-Retrieval System", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/hackrx/run", response_model=SimplifiedHackRXResponse)
async def hackrx_run(request: HackRXRequest):
    """
    HackRX endpoint: Process document from URL and answer questions
    Flow: URL Download -> Document Processing -> Embedding Creation -> Question Answering
    """
    start_time = datetime.now()
    
    try:
        # Step 1: Download and process document from URL
        document_result = await url_document_processor.download_and_process_document(
            request.documents, 
            domain="insurance"  # Default to insurance for policy documents
        )
        
        # Step 2: Create embeddings for the document
        embedding_result = await embedding_search.create_embeddings(
            document_result["text_chunks"], 
            document_result["document_id"]
        )
        
        # Step 3: Process each question
        answer_strings = []
        for question in request.questions:
            try:
                # Parse the question using LLM
                parsed_query = await llm_parser.parse_query(question, "insurance")
                
                # Search for relevant content
                search_results = await embedding_search.search(
                    parsed_query["processed_query"], 
                    "insurance",
                    top_k=5
                )
                
                # Extract relevant text chunks from search results
                relevant_chunks = [result.text for result in search_results if result.similarity_score > 0.3]
                
                # Generate answer using LLM with chunked processing
                llm_result = await llm_parser.generate_answer_from_chunks(
                    question, 
                    relevant_chunks, 
                    "insurance"
                )
                
                # Use the LLM result directly for better answers
                final_answer = llm_result["answer"]
                
                answer_strings.append(final_answer)
                
            except Exception as e:
                # If processing fails for a question, provide error answer
                answer_strings.append(f"Error processing question: {str(e)}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create simplified response with just answers
        return SimplifiedHackRXResponse(answers=answer_strings)
    
    except Exception as e:
        # If document processing fails, return errors for all questions
        error_answers = [f"Failed to process document: {str(e)}" for _ in request.questions]
        return SimplifiedHackRXResponse(answers=error_answers)

@app.post("/hackrx/simple", response_model=SimplifiedHackRXResponse)
async def hackrx_simple(request: HackRXRequest):
    """
    Simplified HackRX endpoint: Returns only the answers as a list of strings
    """
    start_time = datetime.now()
    
    try:
        # Step 1: Download and process document from URL
        document_result = await url_document_processor.download_and_process_document(
            request.documents, 
            domain="insurance"
        )
        
        # Step 2: Create embeddings for the document
        embedding_result = await embedding_search.create_embeddings(
            document_result["text_chunks"], 
            document_result["document_id"]
        )
        
        # Step 3: Process each question and collect answers
        answer_strings = []
        for question in request.questions:
            try:
                # Parse the question using LLM
                parsed_query = await llm_parser.parse_query(question, "insurance")
                
                # Search for relevant content
                search_results = await embedding_search.search(
                    parsed_query["processed_query"], 
                    "insurance",
                    top_k=5
                )
                
                # Extract relevant text chunks from search results
                relevant_chunks = [result.text for result in search_results if result.similarity_score > 0.3]
                
                # Generate answer using LLM with chunked processing
                llm_result = await llm_parser.generate_answer_from_chunks(
                    question, 
                    relevant_chunks, 
                    "insurance"
                )
                
                # Use the LLM result directly for better answers
                final_answer = llm_result["answer"]
                
                answer_strings.append(final_answer)
                
            except Exception as e:
                # If processing fails for a question, provide error answer
                answer_strings.append(f"Error processing question: {str(e)}")
        
        return SimplifiedHackRXResponse(answers=answer_strings)
        
    except Exception as e:
        # If document processing fails, return errors for all questions
        error_answers = [f"Failed to process document: {str(e)}" for _ in request.questions]
        return SimplifiedHackRXResponse(answers=error_answers)

@app.post("/upload", response_model=Dict[str, Any])
async def upload_document(file: UploadFile = File(...), domain: str = Form(...)):
    """
    Upload and process documents (PDF, DOCX, or email)
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx', '.doc', '.eml', '.msg')):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Process document
        result = await document_processor.process_document(file, domain)
        
        # Create embeddings
        embedding_result = await embedding_search.create_embeddings(
            result["text_chunks"], 
            result["document_id"]
        )
        
        return {
            "document_id": result["document_id"],
            "filename": file.filename,
            "domain": domain,
            "chunks_processed": len(result["text_chunks"]),
            "embeddings_created": embedding_result["embeddings_count"],
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process natural language queries with structured decision-making
    Flow: Input -> LLM Parser -> Embedding Search -> Clause Matching -> Logic Evaluation -> JSON Output
    """
    try:
        # Step 1: LLM Parser - Extract structured query
        parsed_query = await llm_parser.parse_query(request.query, request.domain)
        
        # Step 2: Embedding Search - FAISS retrieval
        search_results = await embedding_search.search(
            parsed_query["processed_query"], 
            request.domain,
            top_k=request.top_k or 10
        )
        
        # Step 3: Clause Matching - Semantic similarity
        matched_clauses = await clause_matcher.match_clauses(
            parsed_query, 
            search_results, 
            request.domain
        )
        
        # Step 4: Logic Evaluation - Decision processing
        evaluation_result = await logic_evaluator.evaluate(
            parsed_query, 
            matched_clauses, 
            request.domain
        )
        
        # Step 5: JSON Output - Structured response
        response = QueryResponse(
            query_id=f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            original_query=request.query,
            domain=request.domain,
            parsed_intent=parsed_query["intent"],
            entities=parsed_query["entities"],
            matched_documents=evaluation_result["matched_documents"],
            decision=evaluation_result["decision"],
            confidence_score=evaluation_result["confidence_score"],
            explanation=evaluation_result["explanation"],
            source_references=evaluation_result["source_references"],
            processing_time=evaluation_result["processing_time"]
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Get list of processed documents"""
    try:
        documents = await document_processor.get_documents()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain_decision(query_id: str):
    """Get detailed explanation for a specific decision"""
    try:
        explanation = await logic_evaluator.get_detailed_explanation(query_id)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings"""
    try:
        result = await document_processor.delete_document(document_id)
        await embedding_search.delete_embeddings(document_id)
        return {"status": "success", "message": f"Document {document_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "document_processor": "active",
            "llm_parser": "active",
            "embedding_search": "active",
            "clause_matcher": "active",
            "logic_evaluator": "active"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
