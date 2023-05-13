from typing import List, Optional
from pydantic import BaseModel


# models for query
class Filter(BaseModel):
    document_id: Optional[str] = None
    source: Optional[str] = None
    source_id: Optional[str] = None
    author: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class Query(BaseModel):
    query: str
    filter: Filter
    top_k: int = 3  # TODO need to get it from aii_admin chat settings


class Queries(BaseModel):
    queries: List[Query]


# models for query response
class Metadata(BaseModel):
    source: Optional[str] = None
    source_id: Optional[str] = None
    url: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[str] = None
    document_id: str


class Result(BaseModel):
    id: str
    text: str
    metadata: Metadata
    embedding: List
    score: float


class QueryResult(BaseModel):
    query: str
    results: List[Result]


class ResultModel(BaseModel):
    results: List[QueryResult]
