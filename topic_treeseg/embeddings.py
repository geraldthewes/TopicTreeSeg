from typing import Callable, Any
from pydantic import BaseModel

class Embeddings(BaseModel):
    embeddings_func : Callable[[BaseModel,list],list]
    headers: dict = {}
    model: str
    endpoint: str
