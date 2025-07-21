from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class DomainType(str, Enum):
    INSURANCE = "insurance"
    LEGAL = "legal"
    HR = "hr"
    COMPLIANCE = "compliance"

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    domain: DomainType = Field(..., description="Domain context for the query")
    top_k: Optional[int] = Field(10, description="Number of top results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

class EntityExtraction(BaseModel):
    entity_type: str
    value: str
    confidence: float
    position: tuple

class MatchedDocument(BaseModel):
    document_id: str
    filename: str
    relevance_score: float
    matched_sections: List[str]
    page_numbers: Optional[List[int]] = None

class SourceReference(BaseModel):
    document_id: str
    document_name: str
    section: str
    page_number: Optional[int] = None
    confidence: float
    excerpt: str

class DecisionResult(BaseModel):
    decision_type: str
    outcome: str
    reasoning: List[str]
    risk_level: Optional[str] = None
    recommendations: Optional[List[str]] = None

class QueryResponse(BaseModel):
    query_id: str
    original_query: str
    domain: DomainType
    parsed_intent: str
    entities: List[EntityExtraction]
    matched_documents: List[MatchedDocument]
    decision: DecisionResult
    confidence_score: float = Field(..., ge=0, le=1)
    explanation: str
    source_references: List[SourceReference]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    domain: DomainType
    upload_date: datetime
    file_size: int
    chunks_count: int
    status: str

class ClauseMatch(BaseModel):
    clause_id: str
    clause_text: str
    similarity_score: float
    clause_type: str
    document_source: str
    metadata: Optional[Dict[str, Any]] = None

class ParsedQuery(BaseModel):
    original_query: str
    processed_query: str
    intent: str
    entities: List[EntityExtraction]
    query_type: str
    domain_specific_terms: List[str]
    confidence: float

class EmbeddingResult(BaseModel):
    chunk_id: str
    text: str
    similarity_score: float
    document_id: str
    metadata: Dict[str, Any]

class LogicEvaluationResult(BaseModel):
    matched_documents: List[MatchedDocument]
    decision: DecisionResult
    confidence_score: float
    explanation: str
    source_references: List[SourceReference]
    processing_time: float
    intermediate_steps: List[Dict[str, Any]]

# HackRX specific schemas
class HackRXRequest(BaseModel):
    documents: str = Field(..., description="URL to the document")
    questions: List[str] = Field(..., description="List of questions to ask about the document")

class HackRXAnswer(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[str]
    explanation: str

class HackRXResponse(BaseModel):
    document_id: str
    document_url: str
    processing_time: float
    answers: List[HackRXAnswer]
    status: str

# Simplified response format
class SimplifiedHackRXResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers to the questions")
