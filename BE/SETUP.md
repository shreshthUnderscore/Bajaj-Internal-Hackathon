# Quick Setup Guide

## Prerequisites
- Python 3.8+
- OpenAI API Key

## Installation & Setup

1. **Navigate to the backend directory:**
   ```bash
   cd BE
   ```

2. **Run the startup script:**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

3. **Add your OpenAI API key to `.env`:**
   ```bash
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   ```

4. **Test the system:**
   ```bash
   python test_hackrx.py
   ```

## API Endpoint

**POST** `/hackrx/run`

### Request Format:
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the coverage limit?",
        "What are the exclusions?"
    ]
}
```

### Response Format:
```json
{
    "document_id": "uuid-string",
    "document_url": "https://example.com/document.pdf",
    "processing_time": 15.23,
    "answers": [
        {
            "question": "What is the coverage limit?",
            "answer": "The coverage limit is...",
            "confidence": 0.85,
            "sources": ["Section 3.2: Coverage limits..."],
            "explanation": "Found specific clause about coverage"
        }
    ],
    "status": "success"
}
```

## System Architecture

1. **PDF URL Download** → Downloads document from blob URL
2. **LLM Parser** → Extracts structured information 
3. **FAISS Embedding Search** → Creates and searches vector embeddings
4. **Clause Matching** → Identifies relevant document sections
5. **Logic Evaluation** → Processes information with domain knowledge
6. **JSON Response** → Returns structured answers with explanations

## Features

- ✅ **Multi-format Support**: PDF, DOCX, email documents
- ✅ **URL Processing**: Direct document download from URLs
- ✅ **FAISS Vector Search**: Fast semantic similarity search
- ✅ **Domain Expertise**: Specialized for insurance, legal, HR, compliance
- ✅ **Explainable AI**: Detailed reasoning and source attribution
- ✅ **Structured Output**: JSON responses with confidence scores

## Testing

The system includes a comprehensive test script that validates the complete flow:

```bash
python test_hackrx.py
```

This will test the exact payload format specified in your requirements.
