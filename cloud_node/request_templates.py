from pydantic import BaseModel


class InitProcessRequest(BaseModel):
    start_date: str
    end_date: str
    is_cache_active: bool
    genetic_evaluation_strategy: str
    model_type: str
