# src/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.query import QueryProcessor
from llm.llm_client import LLMClient

app = FastAPI()

class Query(BaseModel):
    text: str
    filters: Optional[dict] = None

class Response(BaseModel):
    answer: str
    confidence: float
    sources: List[dict]

# Initialize components
query_processor = QueryProcessor()
llm_client = LLMClient()

@app.post("/query")
async def process_query(query: Query):
    try:
        # Process query
        results = query_processor.process_query(query.text)
        
        # Generate response
        response = llm_client.generate_response(
            query=query.text,
            context=results['results']['documents'][0]
        )
        
        return Response(
            answer=response['text'],
            confidence=response['confidence'],
            sources=[{"title": m["title"], "url": m["url"]} 
                    for m in results['results']['metadatas'][0]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))