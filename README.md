# LLM-Powered Intelligent Query-Retrieval System

## 🎯 Overview
An advanced document processing and query retrieval system designed for insurance, legal, HR, and compliance domains. The system uses state-of-the-art embeddings and LLM technology to provide intelligent, contextual responses with explainable decision rationale.

## 🚀 Key Features
- **Multi-format Document Processing**: PDFs, DOCX, emails
- **URL-based Document Processing**: Direct download from blob URLs
- **Semantic Search**: FAISS embeddings for intelligent retrieval
- **Clause Matching**: Advanced pattern recognition for legal/policy documents
- **Explainable AI**: Detailed decision rationale and source attribution
- **Structured Responses**: JSON-formatted outputs with confidence scores
- **Domain-specific Intelligence**: Optimized for insurance, legal, HR, and compliance

## 🏗️ Architecture Flow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Blob URL  │───▶│   LLM Parser    │───▶│ Embedding Search│
│                 │    │                 │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Document Download│    │ Clause Matching │    │ Logic Evaluation│
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │  JSON Output    │
                    │ (Structured)    │
                    └─────────────────┘
```

## 🛠️ Tech Stack
- **Backend**: FastAPI, Python
- **Vector Database**: FAISS
- **Document Processing**: PyPDF2, python-docx, email parser
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-4
- **HTTP Client**: aiohttp for document downloading

## 🚦 Main Endpoint

### POST /hackrx/run
**Process document from URL and answer questions**

**Request:**
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=...",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
```

**Response:**
```json
{
    "document_id": "uuid-string",
    "document_url": "https://...",
    "processing_time": 15.23,
    "answers": [
        {
            "question": "What is the grace period for premium payment?",
            "answer": "The grace period for premium payment is 30 days...",
            "confidence": 0.85,
            "sources": ["Section 3.2: Premium payments..."],
            "explanation": "Found specific clause about premium payment"
        }
    ],
    "status": "success"
}
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key

### Setup
```bash
# Navigate to backend directory
cd BE

# Run the startup script (recommended)
./start.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy environment file and add your API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start the server
uvicorn main:app --reload --port 8000
```

### Testing
```bash
# Test the main endpoint
python test_hackrx.py

# Or use curl
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": ["What is the coverage limit?"]
  }'
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Performance
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=50

# Domain-specific thresholds
INSURANCE_CONFIDENCE_THRESHOLD=0.7
LEGAL_CONFIDENCE_THRESHOLD=0.8
```

## 📚 API Documentation
- Interactive docs: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json
- Detailed API guide: [API_DOCUMENTATION.md](BE/API_DOCUMENTATION.md)

## 🧩 System Components

### 1. Document Processor
- Extracts text from PDF, DOCX, email formats
- Chunks text for optimal processing
- Handles metadata storage

### 2. URL Document Processor
- Downloads documents from URLs
- Supports various blob storage providers
- Temporary file management

### 3. LLM Parser
- Structures natural language queries
- Extracts entities and intent
- Domain-specific query understanding

### 4. Embedding Search
- Vector-based semantic search
- FAISS index management
- Similarity scoring

### 5. Clause Matcher
- Domain-specific pattern matching
- Legal/policy clause identification
- Semantic similarity calculation

### 6. Logic Evaluator
- Decision engine with explainable reasoning
- Confidence scoring
- Source attribution

## 🎯 Domain Capabilities

### Insurance
- Policy analysis and interpretation
- Claim processing automation
- Risk assessment
- Coverage determination

### Legal
- Contract review and analysis
- Clause extraction and matching
- Compliance checking
- Legal document summarization

### HR
- Policy interpretation
- Employee handbook queries
- Benefits analysis
- Compliance verification

### Compliance
- Regulatory analysis
- Audit support
- Requirement verification
- Violation detection

## 🔒 Security Features
- Input validation and sanitization
- Secure file handling
- API rate limiting
- Error handling and logging

## 📊 Performance
- Optimized embedding models
- Efficient vector search
- Chunked document processing
- Async/await for concurrent operations

## 🛡️ Error Handling
- Graceful degradation
- Detailed error messages
- Partial result returns
- Comprehensive logging

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License
This project is licensed under the MIT License.

## 📞 Support
For questions or issues, please check the documentation or create an issue in the repository.