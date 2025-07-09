"""FastAPI application for the MOSDAC AI Help Bot API."""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import logging
import uvicorn

from .schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    IntentType,
    ErrorResponse
)
from src.vector_search import search, initialize_retrieval
from src.nlp.enhanced_intent_classifier import MOSDACIntentClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MOSDAC AI Help Bot API",
    description="API for the MOSDAC AI Help Bot with advanced retrieval and NLP capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
intent_classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize components when the application starts."""
    global intent_classifier
    try:
        logger.info("Initializing retrieval components...")
        initialize_retrieval()
        
        logger.info("Initializing intent classifier...")
        intent_classifier = MOSDACIntentClassifier()
        
        logger.info("API startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")
        raise

@app.post("/search", response_model=SearchResponse, responses={
    200: {"model": SearchResponse},
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def search_endpoint(request: SearchRequest):
    """
    Search the MOSDAC knowledge base with natural language queries.
    
    - **query**: The search query string
    - **top_k**: Number of results to return (default: 5)
    - **filters**: Optional filters to apply to the search
    """
    try:
        # Classify the query intent
        intent_result = intent_classifier.predict_intent(request.query)
        
        # Perform the search
        search_results = search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters or {}
        )
        
        # Format the response
        response = SearchResponse(
            query=request.query,
            processed_query=search_results[0].get('metadata', {}).get('processed_query', request.query) if search_results else request.query,
            intent=IntentType(intent_result['intent']),
            confidence=float(intent_result['confidence']),
            results=[
                SearchResult(
                    text=result['text'],
                    score=result['score'],
                    metadata=result.get('metadata', {}),
                    intent=IntentType(intent_result['intent']),
                    confidence=float(intent_result['confidence'])
                )
                for result in search_results
            ],
            metadata={
                'intent_explanation': intent_result.get('explanation', ''),
                'search_metrics': {
                    'total_results': len(search_results)
                }
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Search failed", "details": str(e)}
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
