from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    checkpoint_ns: Optional[str] = None
    checkpoint_id: Optional[str] = None
