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
                elif operation == "initialize_cloud_pretraining":
                    response = init_pretraining_process(data)
                elif operation == "initialize_cloud_training":
                    response = init_periodical_process(data)
                elif operation == "get_cloud_service_state":
                    response = get_cloud_service_state()
                elif operation == "get_federated_simulation_state":
                    response = get_federated_simulation_state()
                elif operation == "get_training_process_parameters":
                    response = get_training_process_parameters()
                elif operation == "perform_model_evaluation":
                    response = perform_model_evaluation(data)
                else:
                    response = {"error": "Invalid operation"}
            except Exception as e:
                response = {"error": str(e)}

            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")


def get_cloud_status():
    """
    :return: retrieves the status of the cloud node
    """
    try:
        return CloudService.get_status()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def get_cloud_service_state():
    return CloudService.get_cloud_service_state()


def get_federated_simulation_state():
    return CloudService.get_federated_simulation_state()


def get_training_process_parameters():
    return CloudService.get_training_process_parameters()


def init_pretraining_process(data):
    try:
        logger.info(f"Pretraining Federation-> Received data: start_date: {data.get('start_date')}, end_date: "
                    f"{data.get('end_date')}, is_cache_active: {data.get('is_cache_active')}, "
                    f"genetic evaluation strategy: {data.get('genetic_evaluation_strategy')}, model type: "
                    f"{data.get('model_type')}")
        CloudService.execute_training_process(
            data.get("start_date"),
            data.get("end_date"),
            data.get("is_cache_active"),
            data.get("genetic_evaluation_strategy"),
            data.get("model_type")
        )
        return {
            "message": "Cloud pretraining process has been started."
        }
    except ValueError as e:
        logger.error(f"Error in init_pretraining_process: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unhandled error in init_pretraining_process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def init_periodical_process(data):
    try:
        logger.info(f"Daily Federation-> Received data: start_date: {data.get('start_date')}, end_date: "
                    f"{data.get('end_date')}, is_cache_active: {data.get('is_cache_active')}, "
                    f"genetic evaluation strategy: {data.get('genetic_evaluation_strategy')}, model type: "
                    f"{data.get('model_type')}")
        CloudService.execute_training_process(
            None,
            data.get("end_date"),
            data.get("is_cache_active"),
            data.get("genetic_evaluation_strategy"),
            data.get("model_type")
        )
        return {
            "message": "Cloud periodical process has been started."
        }
    except ValueError as e:
        logger.error(f"Error in init_periodical_process: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unhandled error in init_periodical_process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def perform_model_evaluation(data):
    CloudService.perform_model_evaluation(data.get("end_date"))
    return {"message": "Model evaluation has been started."}

