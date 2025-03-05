# shared/utils.py

import json
import os
import aio_pika

from shared.logging_config import logger

metric_weights = {
    "mse": 0.10,      # Mean Squared Error: a standard error measure
    "logcosh": 0.15,  # Log-Cosh loss: slightly higher weight because it’s robust to outliers
    "huber": 0.10,    # Huber's loss: similar to MSE but less sensitive to outliers
    "msle": 0.05,     # Mean Squared Log Error: useful if you care about relative differences
    "mae": 0.20,      # Mean Absolute Error: often very interpretable, so could be weighted more
    "r2": -0.35       # R2: negative weight because a higher R² is better, so subtracting it helps lower the overall
                      # score
}


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
