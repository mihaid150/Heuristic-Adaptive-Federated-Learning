# edge_node/edge_service.py
import pika
import json
import base64
import os
import asyncio
import time
import threading
import requests
from pika.exceptions import AMQPConnectionError
from shared.fed_node.node_state import NodeState
from shared.fed_node.fed_node import MessageScope, FedNode, ParentFedNode, get_mac_address
from edge_node.edge_resources_paths import EdgeResourcesPaths
from edge_node.model_manager import pretrain_edge_model, retrain_edge_model, evaluate_edge_model, validate_data_size
from shared.monitoring_thread import MonitoringThread
from shared.utils import delete_all_files_in_folder, publish_message
from shared.logging_config import logger
from shared.utils import metric_weights
from shared.system_metric_collector import SystemMetricCollector


class EdgeService:
    FOG_RABBITMQ_HOST = "fog-rabbitmq-host"
    FOG_EDGE_RECEIVE_QUEUE = "fog_to_edge_"
    FOG_EDGE_RECEIVE_EXCHANGE = "fog_to_edge_exchange"
    EDGE_FOG_SEND_QUEUE = "edge_to_fog_queue"
    rabbitmq_init_monitor_thread = None
    fog_listener_thread = None
    _publisher_loop = None
    sequence_length = 144
    system_metrics_collector = SystemMetricCollector(sampling_rate=2.0)  # measure every two seconds

    completed_previous_round_file_path = os.path.join(
        EdgeResourcesPaths.FLAGS_FOLDER_PATH,
        EdgeResourcesPaths.COMPLETED_PREVIOUS_ROUND_JSON_FILE
    )

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
                        EdgeService.set_completed_previous_round_flag(False)
                        EdgeService.execute_message_instructions(message)
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
    def execute_message_instructions(message):
        """
        Process a received message from the fog node (via RabbitMQ).
        It uses the common processing logic and then publishes the resulting response
        to the fixed EDGE_FOG_SEND_QUEUE.
        """
        try:
            # Process the incoming message using the shared evaluation logic.
            response = EdgeService._process_message(message)
            if response.get("scope") == MessageScope.TEST_DATA_ENOUGH_EXISTS.value:
                logger.info(f"Publishing to fixed queue {EdgeService.EDGE_FOG_SEND_QUEUE} response -> edge_id: "
                            f"{response['edge_id']}, enough_data_existence: {response['enough_data_existence']}.")
            else:
                logger.info(f"Publishing to fixed queue {EdgeService.EDGE_FOG_SEND_QUEUE} response -> edge_id: "
                            f"{response['edge_id']}, metrics: {response['metrics']}.")
            # Publish response to the fixed queue
            EdgeService.publish_response(response)
            EdgeService.set_completed_previous_round_flag(True)
        except Exception as e:
            logger.error(f"Error processing the received message: {str(e)}")

    @staticmethod
    def execute_model_training_evaluation_http(message: dict) -> dict:
        """
        Process a received message (via HTTP) containing the payload and model file,
        and return a JSON response with the processing results.
        """
        return EdgeService._process_message(message)

    @staticmethod
    def _process_message(message: dict) -> dict:
        logger.info(
            f"Received message logging: {message.get('scope')}, start date {message.get('start_date')}, "
            f"current date {message.get('current_date')}, learning rate {message.get('learning_rate')}, "
            f"batch size {message.get('batch_size')}, epochs {message.get('epochs')}, "
            f"patience {message.get('patience')}, fine tune {message.get('fine_tune_layers')}"
        )
        scope = int(message.get("scope"))
        start_date = message.get("start_date")
        current_date = message.get("current_date")

        if scope == MessageScope.TEST_DATA_ENOUGH_EXISTS.value:
            is_training_validation = message.get("is_training_validation")
            response = {
                "edge_id": NodeState.get_current_node().id,
                "edge_mac": get_mac_address(),
                "scope": MessageScope.TEST_DATA_ENOUGH_EXISTS.value,
                "enough_data_existence": validate_data_size(start_date, current_date, EdgeService.sequence_length,
                                                            is_training_validation)
            }
        else:
            try:
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
                    EdgeService.system_metrics_collector.start()
                    metrics = pretrain_edge_model(
                        edge_model_file_path, start_date, current_date,
                        learning_rate, batch_size, epochs, patience, fine_tune_layers, EdgeService.sequence_length
                    )
                    EdgeService.system_metrics_collector.stop()
                    system_metrics = EdgeService.system_metrics_collector.get_average_metrics()

                    pretrained_model_file_path = os.path.join(
                        EdgeResourcesPaths.MODELS_FOLDER_PATH,
                        EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME
                    )
                    with open(pretrained_model_file_path, "rb") as model_file:
                        model_bytes = model_file.read()
                        model_file_base64_resp = base64.b64encode(model_bytes).decode("utf-8")
                    response = {
                        "edge_id": NodeState.get_current_node().id,
                        "edge_mac": get_mac_address(),
                        "metrics": metrics,
                        "system_metrics": system_metrics,
                        "model_file": model_file_base64_resp,
                        "scope": MessageScope.TRAINING.value
                    }
                else:
                    if scope == MessageScope.EVALUATION.value:
                        EdgeService.system_metrics_collector.start()

                        eval_seq_length = EdgeService.sequence_length // 2

                        metrics, prediction_pairs = evaluate_edge_model(edge_model_file_path, current_date,
                                                                        EdgeService.sequence_length)
                        EdgeService.system_metrics_collector.stop()
                        system_metrics = EdgeService.system_metrics_collector.get_average_metrics()

                        response = {
                            "edge_id": NodeState.get_current_node().id,
                            "edge_mac": get_mac_address(),
                            "metrics": metrics,
                            "system_metrics": system_metrics,
                            "prediction_pairs": prediction_pairs,
                            "scope": MessageScope.EVALUATION.value,
                            "evaluation_date": current_date
                        }
                    else:
                        # Retraining process.
                        EdgeService.system_metrics_collector.start()
                        metrics = retrain_edge_model(
                            edge_model_file_path, current_date,
                            learning_rate, batch_size, epochs, patience, fine_tune_layers, EdgeService.sequence_length
                        )
                        EdgeService.system_metrics_collector.stop()
                        system_metrics = EdgeService.system_metrics_collector.get_average_metrics()

                        def compute_weighted_score(metrics_for_score, weights_for_score):
                            return sum(
                                weight * metrics_for_score[metric] for metric, weight in weights_for_score.items())

                        score_before = compute_weighted_score(metrics["before_training"], metric_weights)
                        score_after = compute_weighted_score(metrics["after_training"], metric_weights)

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
                                "edge_mac": get_mac_address(),
                                "metrics": metrics,
                                "system_metrics": system_metrics,
                                "model_file": model_file_base64_resp,
                                "scope": MessageScope.TRAINING.value
                            }
                        else:
                            with open(edge_model_file_path, "rb") as model_file:
                                model_bytes = model_file.read()
                                model_file_base64_resp = base64.b64encode(model_bytes).decode("utf-8")
                            response = {
                                "edge_id": NodeState.get_current_node().id,
                                "edge_mac": get_mac_address(),
                                "metrics": metrics,
                                "system_metrics": system_metrics,
                                "model_file": model_file_base64_resp,
                                "scope": MessageScope.TRAINING.value
                            }
            except Exception as e:
                logger.warning(f"There was met an error in process message: {e}")
                response = {"error": str(e)}
        # (Consider moving file cleanup to a point after the response has been fully handled)
        delete_all_files_in_folder(EdgeResourcesPaths.MODELS_FOLDER_PATH, filter_string=None)
        delete_all_files_in_folder(EdgeResourcesPaths.FILTERED_DATA_FOLDER_PATH, filter_string=None)
        return response

    @staticmethod
    def set_edge_data_parameters(data):
        sequence_length = int(data["sequence_length"])
        EdgeService.sequence_length = sequence_length

        return {
            "message": "The edge data parameters configured successfully.",
        }

    @staticmethod
    def get_edge_training_parameters():
        return {
            "sequence_length": EdgeService.sequence_length
        }

    @staticmethod
    def get_completed_previous_round_flag():
        logger.info("Getting completed previous round flag:")
        if os.path.exists(EdgeService.completed_previous_round_file_path):
            logger.info("Flag file exists.")
            with open(EdgeService.completed_previous_round_file_path, 'r') as file:
                data = json.load(file)
                flag = bool(data.get('completed_previous_round_flag', False))
                logger.info(f"The value of the flag {flag}.")
                return flag
        else:
            return True

    @staticmethod
    def set_completed_previous_round_flag(flag: bool):
        logger.info(f"Completed previous round flag is set now as : {flag}.")
        data = {
            'completed_previous_round_flag': flag,
            'id': NodeState.get_current_node().id,
            'name': NodeState.get_current_node().name,
            'fed_node_type': NodeState.get_current_node().fed_node_type,
            'ip_address': NodeState.get_current_node().ip_address,
            'port': NodeState.get_current_node().port,
            'parent_node_id': NodeState.get_current_node().parent_node.id,
            'parent_node_name': NodeState.get_current_node().parent_node.name,
            'parent_node_fed_node_type': NodeState.get_current_node().parent_node.fed_node_type,
            'parent_node_ip_address': NodeState.get_current_node().parent_node.ip_address,
            'parent_node_port': NodeState.get_current_node().parent_node.port
            }

        with open(EdgeService.completed_previous_round_file_path, 'w') as file:
            json.dump(data, file)

    @staticmethod
    def execute_fog_notification_about_not_finished_training_round():
        logger.info("Executing fog notification about finished training round.")
        with open(EdgeService.completed_previous_round_file_path, 'r') as file:
            data = json.load(file)
        logger.info(f"The data extracted from completed previous round file: {data}.")
        current_node = FedNode(data.get("id"), data.get("name"), data.get("fed_node_type"), data.get("ip_address"),
                               data.get("port"))
        NodeState.initialize_node(current_node)

        logger.info(f"Initialized current node as {NodeState.get_current_node()}")

        parent_node = ParentFedNode(data.get("parent_node_id"), data.get("parent_node_name"),
                                    data.get("parent_node_fed_node_type"), data.get("parent_node_ip_address"),
                                    data.get("parent_node_port"))
        current_node.set_parent_node(parent_node)

        logger.info(f"The parent of the current node is {parent_node}.")

        time.sleep(300)

        url = (f"http://{parent_node.ip_address}:{parent_node.port}/fog/"
               f"notify-fog-from-edge-about-not-completed-previous-round")
        try:
            r = requests.post(url, json={"edge_id": current_node.id}, timeout=None)
            logger.info(f"Received HTTP response from {url}: status code {r.status_code}")
            if 200 <= r.status_code < 300:
                logger.info(f"Response from {url} accepted.")
            else:
                logger.error(f"HTTP error from {url}: status code {r.status_code}")
        except Exception as e1:
            logger.error(f"HTTP request error to {url}: {e1}")