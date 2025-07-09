"""API request/response schemas for the MOSDAC AI Help Bot."""
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class IntentType(str, Enum):
    """Supported intent types for the MOSDAC AI Help Bot."""
    DATA_AVAILABILITY = "data_availability"
    DOWNLOAD = "download"
    DOCUMENTATION = "documentation"
    TECHNICAL_SPEC = "technical_spec"
    SUPPORT = "support"
    WEATHER = "weather"
    CLIMATE = "climate"
    OCEAN = "ocean"
    LAND = "land"
    ATMOSPHERE = "atmosphere"
    SATELLITE = "satellite"
    INSTRUMENT = "instrument"
    PARAMETER = "parameter"
    GENERAL = "general"
    CONTACT_SUPPORT = "contact_support"

class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="The search query string")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional filters to apply to the search"
    )

class SearchResult(BaseModel):
    """Model for a single search result."""
    text: str = Field(..., description="The result text content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the result"
    )
    intent: IntentType = Field(..., description="Detected intent of the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")

class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: List[SearchResult] = Field(..., description="List of search results")
    query: str = Field(..., description="The original query")
    processed_query: str = Field(..., description="The processed query after expansion")
    intent: IntentType = Field(..., description="Detected intent of the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the search"
    )

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
