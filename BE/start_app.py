#!/usr/bin/env python3
"""
Startup script for Bajaj Hackathon Backend
Handles missing dependencies gracefully for Render deployment
"""

import sys
import os
import traceback
from typing import Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_and_install_fallback_dependencies():
    """Check for missing dependencies and provide fallbacks"""
    missing_deps = []
    
    # Check critical dependencies
    try:
        import faiss
    except ImportError:
        missing_deps.append("faiss-cpu")
        print("Warning: FAISS not available, using in-memory search fallback")
    
    try:
        import magic
    except ImportError:
        missing_deps.append("python-magic")
        print("Warning: python-magic not available, using basic file type detection")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Application will continue with limited functionality")
    
    return len(missing_deps) == 0

def start_application():
    """Start the FastAPI application"""
    try:
        # Check dependencies
        check_and_install_fallback_dependencies()
        
        # Load environment variables
        from dotenv import load_dotenv
        
        # Try production env first, then development
        if os.path.exists(".env.production"):
            load_dotenv(".env.production")
            print("Loaded production environment variables")
        else:
            load_dotenv()
            print("Loaded development environment variables")
        
        # Import and start the main application
        from main import app
        import uvicorn
        
        # Get configuration from environment
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
        reload = os.getenv("API_RELOAD", "False").lower() == "true"
        
        print(f"Starting server on {host}:{port}")
        print(f"Reload mode: {reload}")
        print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            access_log=True,
            log_level="info"
        )
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_application()
