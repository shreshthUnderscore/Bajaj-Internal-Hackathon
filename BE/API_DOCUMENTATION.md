# LLM-Powered Intelligent Query-Retrieval System API

## Overview
This system processes documents and answers questions using advanced AI techniques including embeddings, semantic search, and large language models.

## Architecture Flow
```
1. PDF Blob URL → Download Document
2. LLM Parser → Extract structured query  
3. Embedding Search → FAISS retrieval
4. Clause Matching → Semantic similarity
5. Logic Evaluation → Decision processing
6. JSON Output → Structured response
```

## Main Endpoint

### POST /hackrx/run

**Description**: Process a document from URL and answer multiple questions

**Request**:
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
```

**Response**:
```json
{
    "document_id": "uuid-string",
    "document_url": "https://example.com/document.pdf",
    "processing_time": 15.23,
    "answers": [
        {
            "question": "What is the grace period for premium payment?",
            "answer": "The grace period for premium payment is 30 days from the due date...",
            "confidence": 0.85,
            "sources": ["Section 3.2: Premium payments must be made within 30 days..."],
            "explanation": "Found specific clause about premium payment grace period"
        }
    ],
    "status": "success"
}
```

## Additional Endpoints

### GET /
Health check endpoint

### GET /health
Detailed system health status

### POST /upload
Upload document files directly (PDF, DOCX, email)

### POST /query
Process single natural language query

### GET /documents
List all processed documents

## Environment Setup

1. Copy `.env.example` to `.env`
2. Set required API keys:
   - `OPENAI_API_KEY` (required)

## Quick Start

```bash
# Navigate to backend directory
cd BE

# Run startup script
./start.sh

# Or manual setup:
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Testing

```bash
# Test the endpoint
python test_hackrx.py

# Or use curl
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": ["What is the coverage limit?"]
  }'
```

## Features

- **Multi-format Support**: PDF, DOCX, email documents
- **URL Processing**: Direct document download from URLs
- **Semantic Search**: FAISS-based vector similarity search
- **Clause Matching**: Advanced pattern recognition for legal/policy documents
- **Explainable AI**: Detailed reasoning and source attribution
- **Domain Expertise**: Specialized for insurance, legal, HR, and compliance
- **Structured Output**: JSON responses with confidence scores

## Architecture Components

1. **Document Processor**: Extracts text from various formats
2. **LLM Parser**: Structures natural language queries
3. **Embedding Search**: Vector-based semantic search using sentence transformers
4. **Clause Matcher**: Domain-specific pattern matching
5. **Logic Evaluator**: Decision engine with explainable reasoning

## Error Handling

The system provides graceful error handling:
- Network errors during document download
- Unsupported file formats
- API rate limiting
- Processing timeouts

All errors return structured responses with error details and partial results where possible.
