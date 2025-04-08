from typing import Any, Dict

from fog_node.fog_service import FogService
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from shared.logging_config import logger

fog_router = APIRouter()


@fog_router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message: Dict[str, Any] = await websocket.receive_json()
            operation: str = message.get('operation')
            data: Dict[str, Any] = message.get('data', {})

            try:
                if operation == "configure_genetic_engine":
                    response = configure_genetic_engine(data)
                elif operation == "get_genetic_engine_configuration":
                    response = get_genetic_engine_configuration()
                elif operation == "configure_training_parameters":
                    response = configure_training_parameters(data)
                elif operation == "get_current_training_parameter_bounds":
                    response = get_current_training_parameter_bounds()
                elif operation == "get_fog_service_state":
                    response = get_fog_service_state()
                else:
                    response = {"error": "Invalid operation"}
            except Exception as e:
                response = {"error": str(e)}

            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.warning("Websocket disconnected.")


@fog_router.post("/notify-fog-from-edge-about-not-completed-previous-round")
async def execute_model_evaluation_endpoint(message: dict):
    FogService.handle_edge_node_unfinished_previous_round(message.get("edge_id"))


def get_fog_service_state():
    return FogService.get_fog_service_state()


def configure_genetic_engine(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts genetic engine parameters from the incoming data and configures
    the genetic engine via FogService.

    Expected keys in `data` are:
      - "population_size"
      - "number_of_generations"
      - "stagnation_limit"

    Each value is cast to int if provided; if not provided, None is passed.
    """
    def to_int_if_possible(val):
        try:
            return int(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    population_size = to_int_if_possible(data.get("population_size"))
    number_of_generations = to_int_if_possible(data.get("number_of_generations"))
    stagnation_limit = to_int_if_possible(data.get("stagnation_limit"))

    FogService.set_genetic_parameters(population_size, number_of_generations, stagnation_limit)
    return {"status": "Genetic engine configured successfully."}


def get_genetic_engine_configuration():
    return FogService.get_genetic_engine_parameters()


def configure_training_parameters(data):
    FogService.set_training_parameters(data.get("learning_rate_lower_bound"), data.get("learning_rate_upper_bound"),
                                       data.get("batch_size_lower_bound"), data.get("batch_size_upper_bound"),
                                       data.get("epochs_lower_bound"), data.get("epochs_upper_bound"),
                                       data.get("patience_lower_bound"), data.get("patience_upper_bound"),
                                       data.get("fine_tune_layers_lower_bound"), data.get("fine_tune_layers_upper_bound"
                                                                                          )
                                       )
    return {"status": "Configure the training parameters."}


def get_current_training_parameter_bounds():
    return FogService.get_training_parameters()
