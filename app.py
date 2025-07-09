from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import networkx as nx
from typing import Dict, List, Optional, Any
import uvicorn
import sys
import os

app = FastAPI(title="MOSDAC AI Help Bot")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Models
class Query(BaseModel):
    query: str

class QueryResponse(BaseModel):
    results: List[Dict[str, str]]
    query_info: Optional[Dict[str, Any]] = None

# Routes
@app.get("/")
async def read_root():
    """Serve the chat widget HTML"""
    return FileResponse("index.html")

@app.post("/query")
async def process_query(query: Query) -> QueryResponse:
    """Process a chat query and return response with knowledge graph"""
    try:
        # TODO: Replace with actual AI processing
        # This is a mock response for demonstration
        bot_reply = f"This is a mock response to: {query.query}"
        
        # Mock entities for knowledge graph
        entities = {
            "topic": ["MOSDAC", "satellite"],
            "data": ["temperature", "rainfall"]
        }
        
        return QueryResponse(
            results=[{"text": bot_reply}],
            query_info={"entities": entities}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server():
    """Run the server with platform-specific settings"""
    if sys.platform == 'win32':
        # Windows-specific configuration
        uvicorn.run(
            app,
            host="127.0.0.1",  # Use localhost instead of 0.0.0.0 on Windows
            port=8000,
            reload=False  # Disable reload on Windows to avoid signal handling issues
        )
    else:
        # Unix-like systems configuration
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )

# Run the server
if __name__ == "__main__":
    run_server() 