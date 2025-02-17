# cloudnode/cloud_main.py

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from cloud_node.cloud_service import CloudService
from shared.logging_config import logger

cloud_router = APIRouter()


@cloud_router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message: Dict[str, Any] = await websocket.receive_json()
            operation: str = message.get('operation')
            data: Dict[str, Any] = message.get('data', {})

            try:
                if operation == "get_cloud_status":
                    response = get_cloud_status()
                elif operation == "initialize_cloud_process":
                    response = init_cloud_process(data)
                else:
                    response = {"error": "Invalid operation"}
            except Exception as e:
                response = {"error": str(e)}

            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.error("WebSocket disconnected.")


def get_cloud_status():
    """
    :return: retrieves the status of the cloud node
    """
    try:
        return CloudService.get_status()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def init_cloud_process(data):
    try:
        logger.info(f"Received data: start_date: {data.get('start_date')}, end_date: {data.get('end_date')},"
                    f" is_cache_active: {data.get('is_cache_active')}, genetic evaluation strategy: "
                    f"{data.get('genetic_evaluation_strategy')}, model type: {data.get('model_type')}")
        CloudService.init_process(
            data.get("start_date"),
            data.get("end_date"),
            data.get("is_cache_active"),
            data.get("genetic_evaluation_strategy"),
            data.get("model_type")
        )
        return {
            "message": "Cloud initial process run successful."
        }
    except ValueError as e:
        logger.error("Error in init_cloud_process:", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Unhandled error in init_cloud_process:", e)
        raise HTTPException(status_code=500, detail=str(e))
