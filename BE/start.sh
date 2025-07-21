#!/bin/bash

echo "🚀 Starting LLM-Powered Intelligent Query-Retrieval System"
echo "=================================================="

# Check if we're in the BE directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the BE directory"
    echo "Usage: cd BE && ./start.sh"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads
mkdir -p logs

# Set environment variables if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please update .env file with your API keys:"
    echo "   - OPENAI_API_KEY (required for LLM processing)"
    echo ""
    echo "🔑 To set your OpenAI API key:"
    echo "   export OPENAI_API_KEY='your-gpt-4o-mini-api-key-here'"
    echo "   or edit the .env file directly"
    echo ""
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ] && ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    echo "⚠️  Warning: OPENAI_API_KEY not found in environment or .env file"
    echo "   The system will not work without a valid OpenAI API key"
    echo ""
fi

# Start the server
echo ""
echo "🧪 Running system health check..."

# Test import of main modules
python3 -c "
import sys
try:
    from fastapi import FastAPI
    from services.document_processor import DocumentProcessor
    from services.llm_parser import LLMParser
    from services.embedding_search import EmbeddingSearch
    from services.clause_matcher import ClauseMatcher
    from services.logic_evaluator import LogicEvaluator
    from services.url_document_processor import URLDocumentProcessor
    print('✅ All modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" || exit 1

echo "✅ System health check passed"
echo ""

# Start the server
echo "🌟 Starting FastAPI server..."
echo "📡 Server will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Interactive API: http://localhost:8000/redoc"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

uvicorn main:app --reload --host 0.0.0.0 --port 8000
