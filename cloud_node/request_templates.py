from pydantic import BaseModel


class InitProcessRequest(BaseModel):
    is_cache_active: bool
    genetic_evaluation_strategy: str
    model_type: str
