# edge_node/edge_main.py
from http.client import HTTPException
from typing import Any, Dict

from fastapi import APIRouter
from starlette.websockets import WebSocket, WebSocketDisconnect
from shared.logging_config import logger
from edge_node.edge_service import EdgeService

edge_router = APIRouter()


@edge_router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message: Dict[str, Any] = await websocket.receive_json()
            operation: str = message['operation']
            data: Dict[str, Any] = message['data']

            try:
                if operation == "configure_edge_data_parameters":
                    response = configure_edge_data_parameters(data)
                elif operation == "get_edge_data_parameters":
                    response = get_edge_data_parameters()
                else:
                    response = {"error": "Invalid operation"}
            except Exception as e:
                response = {"error": str(e)}

            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.warning("Websocket disconnected.")


@edge_router.post("/execute-model-evaluation")
async def execute_model_evaluation_endpoint(message: dict):
    """
    HTTP endpoint to process a model training evaluation message.
    The JSON request body should match the message format.
    """
    result = EdgeService.execute_model_training_evaluation_http(message)
    if "error" in result:
        raise HTTPException()
    return result


def configure_edge_data_parameters(data):
    return EdgeService.set_edge_data_parameters(data)


def get_edge_data_parameters():
    return EdgeService.get_edge_training_parameters()


@edge_router.post("/async-execute-notify-fog-from-edge-about-not-completed-previous-round")
def async_execute_notify_fog_from_edge_about_not_completed_previous_round():
    EdgeService.execute_fog_notification_about_not_finished_training_round()
