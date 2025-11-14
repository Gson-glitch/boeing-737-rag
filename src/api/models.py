from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question about Boeing 737 operations"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the first action after positive rate of climb?"
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    answer: str = Field(
        ...,
        description="Generated answer based on the manual"
    )
    
    pages: List[int] = Field(
        ...,
        description="Page numbers referenced (1-based PDF index)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "After establishing a positive rate of climb, the first action is to call 'GEAR UP' [Document 1].",
                "pages": [39, 51]
            }
        }