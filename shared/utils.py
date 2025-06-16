# shared/utils.py
import asyncio
import json
import os
import aio_pika
import websockets
import tensorflow as tf
from shared.fed_node.fed_node import FedNode
from shared.logging_config import logger

metric_weights = {
    "mse": 0.10,  # Mean Squared Error: a standard error measure
    "logcosh": 0.15,  # Log-Cosh loss: slightly higher weight because it’s robust to outliers
    "huber": 0.10,  # Huber's loss: similar to MSE but less sensitive to outliers
    "msle": 0.05,  # Mean Squared Log Error: useful if you care about relative differences
    "mae": 0.20,  # Mean Absolute Error: often very interpretable, so could be weighted more
    "tail_mae": 0.10,  # MAE on the highest 10% of true values to emphasize spikes
    "r2": -0.35  # R2: negative weight because a higher R² is better, so subtracting it helps lower the overall
}

required_columns = [
    'value',  # Raw consumption value
    'value_diff',  # First difference
    # Window 3 (30 minutes)
    'value_rolling_mean_3',
    'value_volatility_3',
    'value_ewm_3',
    # Window 6 (1 hour)
    'value_rolling_mean_6',
    'value_volatility_6',
    'value_ewm_6',
    # Window 12 (2 hours)
    'value_rolling_mean_12',
    'value_volatility_12',
    'value_ewm_12',
    # Window 24 (4 hours)
    'value_rolling_mean_24',
    'value_volatility_24',
    'value_ewm_24',
    'drift_flag',
    'time_since_last_spike'
]


def delete_all_files_in_folder(folder_path: str, filter_string: str | None) -> None:
    """
    Deletes all files in a given folder
    :param filter_string: String to filter files by name. If None, all files will are deleted.
    :param folder_path: Path to the folder to clear.
    """

    if not os.path.exists(folder_path):
        logger.info(f"The folder {folder_path} does not exist!")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if filter_string is not None and filter_string in filename:
                os.remove(file_path)
            elif filter_string is None:
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")

    logger.info(f"Deleted successfully all files in folder {folder_path}.")


async def publish_message(host: str, routing_key: str, message: dict):
    try:
        connection = await aio_pika.connect_robust(host=host)
        async with connection:
            channel = await connection.channel()
            msg = aio_pika.Message(
                body=json.dumps(message).encode("utf-8")
            )
            # Publish to the default exchange using the provided routing key.
            await channel.default_exchange.publish(msg, routing_key=routing_key)
    except Exception as e:
        logger.error(f"Error publishing message: {e}")


def reinitialize_and_set_parent(node: FedNode, parent_node):
    """
    Asynchronously connects via WebSocket to the node's /node/ws endpoint,
    sends an "initialize" message, then a "set_parent" message if cloud_node is provided.
    """
    async def ws_call():
        ws_url = f"ws://{node.ip_address}:{node.port}/node/ws"
        try:
            async with websockets.connect(ws_url) as websocket:
                init_message = {
                    "operation": "initialize",
                    "data": {
                        "id": node.id,
                        "name": node.name,
                        "node_type": node.fed_node_type,
                        "ip_address": node.ip_address,
                        "port": node.port
                    }
                }
                await websocket.send(json.dumps(init_message))
                init_resp = await websocket.recv()
                logger.info(f"WS initialize response from node {node.id}: {init_resp}")
                if parent_node is not None:
                    set_parent_message = {
                        "operation": "set_parent",
                        "data": {
                            "id": parent_node.id,
                            "name": parent_node.name,
                            "node_type": parent_node.fed_node_type,
                            "ip_address": parent_node.ip_address,
                            "port": parent_node.port
                        }
                    }
                    await websocket.send(json.dumps(set_parent_message))
                    parent_resp = await websocket.recv()
                    logger.info(f"WS set_parent response from node {node.id}: {parent_resp}")
                else:
                    logger.error("Cloud node not set; cannot perform set_parent operation.")
        except Exception as ex:
            logger.error(f"Error during WS call to node {node.id}: {ex}")
    asyncio.run(ws_call())
    return


@tf.keras.utils.register_keras_serializable()
class CombineExperts(tf.keras.layers.Layer):
    """Combine normal and spike predictions using a gating value."""

    def call(self, inputs):
        normal, spike, gate = inputs
        return normal * (1 - gate) + spike * gate

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
