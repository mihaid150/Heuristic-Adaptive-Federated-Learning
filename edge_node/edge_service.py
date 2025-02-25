# edge_node/edge_service.py
import pika
import json
import base64
import os
import asyncio
import time
import threading
from pika.exceptions import AMQPConnectionError
from shared.fed_node.node_state import NodeState
from shared.fed_node.fed_node import ModelScope
from edge_node.edge_resources_paths import EdgeResourcesPaths
from edge_node.model_manager import pretrain_edge_model, retrain_edge_model
from shared.monitoring_thread import MonitoringThread
from shared.utils import delete_all_files_in_folder, publish_message
from shared.logging_config import logger


class EdgeService:
    FOG_RABBITMQ_HOST = "fog-rabbitmq-host"
    FOG_EDGE_RECEIVE_QUEUE = "fog_to_edge_"
    FOG_EDGE_RECEIVE_EXCHANGE = "fog_to_edge_exchange"
    EDGE_FOG_SEND_QUEUE = "edge_to_fog_queue"
    rabbitmq_init_monitor_thread = None
    fog_listener_thread = None
    _publisher_loop = None

    @staticmethod
    def init_rabbitmq() -> None:
        """
        Initialize RabbitMQ connection and declare queues.
        """
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=EdgeService.FOG_RABBITMQ_HOST)
        )
        channel = connection.channel()
        channel.exchange_declare(exchange=EdgeService.FOG_EDGE_RECEIVE_EXCHANGE, exchange_type="direct", durable=True)
        EdgeService.FOG_EDGE_RECEIVE_QUEUE += str(NodeState.get_current_node().id)
        channel.queue_declare(queue=EdgeService.FOG_EDGE_RECEIVE_QUEUE, durable=True)
        channel.queue_bind(queue=EdgeService.FOG_EDGE_RECEIVE_QUEUE, exchange=EdgeService.FOG_EDGE_RECEIVE_EXCHANGE,
                           routing_key=NodeState.get_current_node().id)
        channel.queue_declare(queue=EdgeService.EDGE_FOG_SEND_QUEUE, durable=True)
        connection.close()

    @staticmethod
    def monitor_parent_node_and_init_rabbitmq() -> None:
        """
        Monitor the parent node of the current edge node and initialize RabbitMQ connection when a parent is set.
        Also start the listener if not already running.
        """
        current_node = NodeState.get_current_node()
        if current_node and current_node.parent_node:
            logger.info(f"Parent node detected: {current_node.parent_node.name}. Initializing RabbitMQ connection...")
            EdgeService.FOG_RABBITMQ_HOST = current_node.parent_node.ip_address
            EdgeService.init_rabbitmq()
            if EdgeService.rabbitmq_init_monitor_thread:
                EdgeService.rabbitmq_init_monitor_thread.stop()
            # Start the listener thread if not running
            if not EdgeService.fog_listener_thread or not EdgeService.fog_listener_thread.is_alive():
                EdgeService.fog_listener_thread = threading.Thread(
                    target=EdgeService.listen_to_fog_receiving_queue,
                    daemon=True
                )
                EdgeService.fog_listener_thread.start()

    @staticmethod
    def start_monitoring_parent_node() -> None:
        """
        Start a monitoring thread that waits for a parent node to be assigned and initializes RabbitMQ.
        """
        EdgeService.rabbitmq_init_monitor_thread = MonitoringThread(
            target=EdgeService.monitor_parent_node_and_init_rabbitmq,
            sleep_time=2
        )
        EdgeService.rabbitmq_init_monitor_thread.start()

    @staticmethod
    def _start_publisher_loop():
        if EdgeService._publisher_loop is None:
            EdgeService._publisher_loop = asyncio.new_event_loop()

            def run_loop(loop):
                asyncio.set_event_loop(loop)
                loop.run_forever()
            t = threading.Thread(target=run_loop, args=(EdgeService._publisher_loop,), daemon=True)
            t.start()

    @staticmethod
    def init_process() -> None:
        """
        Initiate the edge node process.
        """
        logger.info("Edge node service process has been initiated.")
        EdgeService.start_monitoring_parent_node()

    @staticmethod
    def listen_to_fog_receiving_queue() -> None:
        """
        Continuously listen to the RabbitMQ queue for messages from the fog node.
        In case of any connection or consumption error, wait 5 seconds and reconnect.
        """
        while True:
            connection = None
            try:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=EdgeService.FOG_RABBITMQ_HOST,
                        heartbeat=30
                    )
                )
                channel = connection.channel()

                def callback(ch, method, properties, body):
                    try:
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        # Deserialize the message
                        message = json.loads(body.decode("utf-8"))
                        # Save reply-to and correlation id from the incoming message
                        message["_reply_to"] = properties.reply_to
                        message["_correlation_id"] = properties.correlation_id

                        logger.info(f"Accepting message for child id {message.get('child_id')}.")
                        EdgeService.execute_model_training_evaluation(message)
                    except Exception as e1:
                        logger.error(f"Error processing message: {str(e1)}")
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

                channel.basic_consume(
                    queue=EdgeService.FOG_EDGE_RECEIVE_QUEUE,
                    on_message_callback=callback,
                    auto_ack=False
                )
                logger.info("Listening for messages from the fog...")
                channel.start_consuming()
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.StreamLostError) as e:
                logger.error(f"Connection error in listen_to_fog_receiving_queue: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in listen_to_fog_receiving_queue: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception as e:
                        logger.error(f"An error occurred in listen_to_fog_receiving_queue finally clause: {e}")

    @staticmethod
    def publish_response(response_message):
        """
        Publishes the response message with retry logic.
        This loop will continue until the message is successfully published.
        Since we are using a fixed queue, the response is always sent to EDGE_FOG_SEND_QUEUE.
        """
        while True:
            try:
                EdgeService._start_publisher_loop()
                # Always publish to the fixed queue.
                target_queue = EdgeService.EDGE_FOG_SEND_QUEUE
                future = asyncio.run_coroutine_threadsafe(
                    publish_message(
                        EdgeService.FOG_RABBITMQ_HOST,
                        target_queue,
                        response_message
                    ),
                    EdgeService._publisher_loop
                )
                result = future.result(timeout=10)
                return result
            except Exception as e:
                logger.error(f"Error in publish_response, retrying: {e}")
                time.sleep(3)

    @staticmethod
    def execute_model_training_evaluation(message):
        """
        Process a received message from the fog node (via RabbitMQ).
        It uses the common processing logic and then publishes the resulting response
        to the fixed EDGE_FOG_SEND_QUEUE.
        """
        try:
            # Process the incoming message using the shared evaluation logic.
            response = EdgeService._process_evaluation_message(message)
            logger.info(f"Publishing to fixed queue {EdgeService.EDGE_FOG_SEND_QUEUE} response -> edge_id: "
                        f"{response['edge_id']}, metrics: {response['metrics']}.")
            # Publish response to the fixed queue
            EdgeService.publish_response(response)
        except Exception as e:
            logger.error(f"Error processing the received message: {str(e)}")

    @staticmethod
    def execute_model_training_evaluation_http(message: dict) -> dict:
        """
        Process a received message (via HTTP) containing the payload and model file,
        and return a JSON response with the processing results.
        """
        return EdgeService._process_evaluation_message(message)

    @staticmethod
    def _process_evaluation_message(message: dict) -> dict:
        """
        Common processing logic for evaluation requests.
        This function extracts parameters, decodes and saves the model file,
        runs the pretraining or retraining process, cleans up temporary files,
        and returns a response dictionary.
        """
        try:
            # genetic_evaluation = message.get("genetic_evaluation")
            start_date = message.get("start_date")
            current_date = message.get("current_date")
            # is_cache_active = message.get("is_cache_active")
            # model_type = message.get("model_type")
            scope = int(message.get("scope"))
            model_file_base64 = message.get("model_file")
            learning_rate = message.get("learning_rate")
            batch_size = message.get("batch_size")
            epochs = message.get("epochs")
            patience = message.get("patience")
            fine_tune_layers = message.get("fine_tune_layers")

            if not model_file_base64:
                raise ValueError("Model file is missing in the received message.")

            # Decode and save the model file locally.
            model_file_bytes = base64.b64decode(model_file_base64)
            edge_model_file_path = os.path.join(
                EdgeResourcesPaths.MODELS_FOLDER_PATH,
                EdgeResourcesPaths.EDGE_MODEL_FILE_NAME
            )
            os.makedirs(EdgeResourcesPaths.MODELS_FOLDER_PATH, exist_ok=True)
            with open(edge_model_file_path, "wb") as model_file:
                model_file.write(model_file_bytes)
            logger.info(f"Received fog model saved at: {edge_model_file_path}")

            if start_date is not None:
                # Pretraining process.
                metrics = pretrain_edge_model(
                    edge_model_file_path, start_date, current_date,
                    learning_rate, batch_size, epochs, patience, fine_tune_layers
                )
                pretrained_model_file_path = os.path.join(
                    EdgeResourcesPaths.MODELS_FOLDER_PATH,
                    EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME
                )
                with open(pretrained_model_file_path, "rb") as model_file:
                    model_bytes = model_file.read()
                    model_file_base64_resp = base64.b64encode(model_bytes).decode("utf-8")
                response = {
                    "edge_id": NodeState.get_current_node().id,
                    "metrics": metrics,
                    "model_file": model_file_base64_resp,
                    "scope": ModelScope.TRAINING.value
                }
            else:
                # Retraining process.
                metrics = retrain_edge_model(
                    edge_model_file_path, current_date,
                    learning_rate, batch_size, epochs, patience, fine_tune_layers
                )
                weights = {
                    "loss": 0.4,
                    "mae": 0.3,
                    "mse": 0.1,
                    "rmse": 0.1,
                    "r2": -0.1,
                }

                def compute_weighted_score(metrics_for_score, weights_for_score):
                    return sum(weight * metrics_for_score[metric] for metric, weight in weights_for_score.items())

                score_before = compute_weighted_score(metrics["before_training"], weights)
                score_after = compute_weighted_score(metrics["after_training"], weights)

                if scope == ModelScope.EVALUATION.value:
                    response = {
                        "edge_id": NodeState.get_current_node().id,
                        "metrics": metrics,
                        "scope": ModelScope.EVALUATION.value,
                        "evaluation_date": current_date
                    }
                else:
                    if score_after < score_before:
                        retrained_model_file_path = os.path.join(
                            EdgeResourcesPaths.MODELS_FOLDER_PATH,
                            EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME
                        )
                        with open(retrained_model_file_path, "rb") as model_file:
                            model_bytes = model_file.read()
                            model_file_base64_resp = base64.b64encode(model_bytes).decode("utf-8")
                        response = {
                            "edge_id": NodeState.get_current_node().id,
                            "metrics": metrics,
                            "model_file": model_file_base64_resp,
                            "scope": ModelScope.TRAINING.value
                        }

                    else:
                        with open(edge_model_file_path, "rb") as model_file:
                            model_bytes = model_file.read()
                            model_file_base64_resp = base64.b64encode(model_bytes).decode("utf-8")
                        response = {
                            "edge_id": NodeState.get_current_node().id,
                            "metrics": metrics,
                            "model_file": model_file_base64_resp,
                            "scope": ModelScope.TRAINING.value
                        }
            return response
        finally:
            delete_all_files_in_folder(EdgeResourcesPaths.MODELS_FOLDER_PATH, filter_string=None)
            delete_all_files_in_folder(EdgeResourcesPaths.FILTERED_DATA_FOLDER_PATH, filter_string=None)
