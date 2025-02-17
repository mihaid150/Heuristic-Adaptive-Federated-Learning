# edge_node/edge_main.py
from http.client import HTTPException

from fastapi import APIRouter

from edge_node.edge_service import EdgeService

edge_router = APIRouter()


@edge_router.post("/execute-model-evaluation")
async def execute_model_evaluation_endpoint(message: dict):
    """
    HTTP endpoint to process a model training evaluation message.
    The JSON request body should match the message format.
    """
    result = EdgeService.execute_model_training_evaluation_http(message)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result