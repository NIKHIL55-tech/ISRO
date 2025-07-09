#!/usr/bin/env python3
"""
Run the MOSDAC AI Help Bot API server.

This script starts the FastAPI server with uvicorn.
"""
import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Set up paths
    base_dir = Path(__file__).parent
    env_path = base_dir / ".env"
    
    # Load environment variables if .env exists
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    reload_flag = os.getenv("RELOAD", "true").lower() == "true"
    
    # Start the server
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=reload_flag,
        log_level=log_level,
        workers=int(os.getenv("WORKERS", "1")),
    )
