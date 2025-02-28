import base64
import json
import os
import time
import threading
from enum import Enum

import pika
from pika.exceptions import AMQPConnectionError
from shared.fed_node.node_state import NodeState
from shared.fed_node.fed_node import ModelScope
from shared.shared_resources_paths import SharedResourcesPaths
from cloud_node.model_manager import create_initial_lstm_model, aggregate_fog_models
from cloud_node.cloud_resources_paths import CloudResourcesPaths
from cloud_node.cloud_cooling_scheduler import CloudCoolingScheduler
from shared.utils import delete_all_files_in_folder
from shared.monitoring_thread import MonitoringThread
from shared.logging_config import logger


class CloudServiceState(Enum):
    IDLE = 1
    OPERATIONAL = 2


class FederatedSimulationState(Enum):
    IDLE = 1
    PRETRAINING = 2
    TRAINING = 3


class CloudService:
    CLOUD_RABBITMQ_HOST = "cloud-rabbitmq-host"
    FOG_CLOUD_RECEIVE_QUEUE = "fog_to_cloud_queue"
    CLOUD_TO_FOG_SEND_EXCHANGE = "cloud_to_fog_exchange"

    cloud_cooling_scheduler = CloudCoolingScheduler()
    received_fog_messages = {}
    received_fog_results = []
    evaluation_received_results_counter = 0
    rabbitmq_init_monitor_thread = None
    fog_models_listener_thread = None

    is_cache_active = False
    genetic_strategy = None
    model_type = None
    start_date = None
    current_date = None

    cloud_service_state = CloudServiceState.IDLE
    federated_simulation_state = FederatedSimulationState.IDLE

    @staticmethod
    def get_cloud_service_state():
        return {
            "cloud_service_state": CloudService.cloud_service_state.value
        }

    @staticmethod
    def get_federated_simulation_state():
        return {
            "federated_simulation_state": CloudService.federated_simulation_state.value
        }

    @staticmethod
    def get_training_process_parameters():
        if CloudService.federated_simulation_state == FederatedSimulationState.PRETRAINING:
            return {
                "start_date": CloudService.start_date,
                "current_date": CloudService.current_date,
                "is_cache_active": CloudService.is_cache_active,
                "genetic_strategy": CloudService.genetic_strategy,
                "model_type": CloudService.model_type,
            }
        else:
            return {
                "current_date": CloudService.current_date,
                "is_cache_active": CloudService.is_cache_active,
                "genetic_strategy": CloudService.genetic_strategy,
            }

    @staticmethod
    def monitor_current_node_init() -> None:
        log_interval = 20  # seconds
        last_log_time = 0
        while True:  # Ensure the thread keeps running
            try:
                current_node = NodeState().get_current_node()
                if current_node is not None:
                    logger.info(f"Cloud node detected: {current_node.name}.")
                    CloudService.CLOUD_RABBITMQ_HOST = current_node.ip_address
                    CloudService.init_rabbitmq()
                    if CloudService.rabbitmq_init_monitor_thread:
                        CloudService.rabbitmq_init_monitor_thread.stop()
                    break  # Exit the loop after successful initialization
                else:
                    now = time.time()
                    if now - last_log_time >= log_interval:
                        logger.info("No node detected yet. Retrying...")
                        last_log_time = now
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
            time.sleep(2)  # Sleep before the next check

    @staticmethod
    def stop_fog_models_listener():
        """
        Stop the current fog models listener thread if it is running.
        """
        if CloudService.fog_models_listener_thread is not None and CloudService.fog_models_listener_thread.is_alive():
            logger.info("Stopping current fog models listener thread...")
            # We assume that the listener thread exits when the cooling process stops.
            # Here we force-stop by joining with a timeout.
            try:
                CloudService.fog_models_listener_thread.join(timeout=5)
            except Exception as e:
                logger.error(f"Error while stopping fog models listener thread: {e}")
            CloudService.fog_models_listener_thread = None

    @staticmethod
    def start_monitoring_current_node() -> None:
        logger.info("Starting monitoring thread for node initialization...")
        CloudService.rabbitmq_init_monitor_thread = MonitoringThread(
            target=CloudService.monitor_current_node_init,
            sleep_time=2
        )
        CloudService.rabbitmq_init_monitor_thread.start()

    @staticmethod
    def init_rabbitmq():
        """
        Initialize RabbitMQ connection and declare queues.
        """
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=CloudService.CLOUD_RABBITMQ_HOST))
        channel = connection.channel()

        # declare the queues for sending and receiving messages
        channel.exchange_declare(exchange=CloudService.CLOUD_TO_FOG_SEND_EXCHANGE, exchange_type="direct", durable=True)
        channel.queue_declare(queue=CloudService.FOG_CLOUD_RECEIVE_QUEUE, durable=True)
        connection.close()

    @staticmethod
    def get_status() -> dict:
        """
        :return: the status of the cloud node.
        """
        cloud_node = NodeState.get_current_node()
        if not cloud_node:
            raise ValueError("Current node is not initialized.")
        return {
            "message": f"Cloud Node '{cloud_node.name} is up and running.'"
        }

    @staticmethod
    def start_listener():
        # only start if there's no listener thread running or if it's not alive.
        CloudService.stop_fog_models_listener()
        if CloudService.fog_models_listener_thread is None or not CloudService.fog_models_listener_thread.is_alive():
            CloudService.fog_models_listener_thread = threading.Thread(
                target=CloudService.listen_to_receive_fog_queue,
                daemon=True
            )
            CloudService.fog_models_listener_thread.start()
            logger.info("Started fog_models_listener thread for fog messages.")
        else:
            logger.info("fog_models_listener thread already running; not starting another.")

    @staticmethod
    def execute_training_process(start_date: str | None, end_date: str, is_cache_active: bool,
                                 genetic_evaluation_strategy: str,
                                 model_type: str):
        CloudService.cloud_service_state = CloudServiceState.OPERATIONAL
        CloudService.cloud_cooling_scheduler.reset()
        CloudService.received_fog_messages = {}

        if CloudService.CLOUD_RABBITMQ_HOST == "cloud-rabbitmq-host":
            raise ValueError("RabbitMQ host is not initialized. Please wait for the node to be detected.")

        cloud_model_file_path = os.path.join(CloudResourcesPaths.MODELS_FOLDER_PATH,
                                             CloudResourcesPaths.CLOUD_MODEL_FILE_NAME)

        cache_cloud_model_file_path = os.path.join(SharedResourcesPaths.CACHE_FOLDER_PATH,
                                                   CloudResourcesPaths.CLOUD_MODEL_FILE_NAME)

        if start_date is not None:
            CloudService.federated_simulation_state = FederatedSimulationState.PRETRAINING
            cloud_model = create_initial_lstm_model()
            cloud_model.save(cloud_model_file_path)

            # delete the model performance json from the previous simulation
            model_performance_file_path = os.path.join(SharedResourcesPaths.CACHE_FOLDER_PATH,
                                                       CloudResourcesPaths.MODEL_PERFORMANCE_RESULTS_FILE_NAME)

            if os.path.exists(model_performance_file_path):
                delete_all_files_in_folder(SharedResourcesPaths.CACHE_FOLDER_PATH, 'model_performance')

            CloudService.received_fog_results = []
        else:
            CloudService.federated_simulation_state = FederatedSimulationState.TRAINING

        CloudService.start_date = start_date
        CloudService.current_date = end_date
        CloudService.is_cache_active = is_cache_active
        CloudService.genetic_strategy = genetic_evaluation_strategy
        CloudService.model_type = model_type

        if os.path.exists(cloud_model_file_path):
            logger.info("Cloud model exists in the model folder.")
            with open(cloud_model_file_path, "rb") as model_file:
                model_bytes = model_file.read()
                encoded_model = base64.b64encode(model_bytes).decode('utf-8')
        else:
            logger.info("Cloud model does not exists in the model folder, loading it from cache.")
            with open(cache_cloud_model_file_path, "rb") as model_file:
                model_bytes = model_file.read()
                encoded_model = base64.b64encode(model_bytes).decode('utf-8')

        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "is_cache_active": is_cache_active,
            "genetic_evaluation_strategy": genetic_evaluation_strategy,
            "model_type": model_type,
            "model_file": encoded_model,
            "scope": ModelScope.TRAINING.value
        }

        CloudService.cloud_cooling_scheduler.start_cooling()
        CloudService.start_listener()
        CloudService._send_message_to_children(payload)

    @staticmethod
    def _send_message_to_children(payload):
        """Sends a message to all child nodes."""

        cloud_node = NodeState.get_current_node()
        if not cloud_node:
            raise ValueError("Current node is not initialized. Go back and initialize it first.")

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=CloudService.CLOUD_RABBITMQ_HOST))
        channel = connection.channel()

        logger.info(f"Sending the cloud model to {len(cloud_node.child_nodes)} fog nodes.")

        for child_node in cloud_node.child_nodes:
            routing_key = str(child_node.id)
            message = {
                "child_id": child_node.id,
                **payload,
            }

            channel.basic_publish(exchange=CloudService.CLOUD_TO_FOG_SEND_EXCHANGE, routing_key=routing_key,
                                  body=json.dumps(message))
            logger.info(f"Request sent to exchange for child {child_node.name} with routing key {routing_key}")

        connection.close()

    @staticmethod
    def get_fog_model(message):
        """
        Process messages from the queue whenever they are received.
        """
        try:
            if not CloudService.cloud_cooling_scheduler.is_cloud_cooling_operational():
                logger.info("Cooling process has finished. Ignoring further messages.")
                aggregate_fog_models(CloudService.received_fog_messages)
                delete_all_files_in_folder(CloudResourcesPaths.MODELS_FOLDER_PATH, filter_string="fog")
                CloudService.cloud_service_state = CloudServiceState.IDLE
                return

            child_id = message.get("fog_id")

            # Process the message
            if child_id not in CloudService.received_fog_messages:
                CloudService.process_received_messages(message, child_id)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    @staticmethod
    def process_received_messages(message: dict, child_id: str) -> None:
        fog_model_file_bytes = base64.b64decode(message["model_file"])
        lambda_prev_value = message.get("lambda_prev")

        fog_model_file_name = CloudResourcesPaths.FOG_MODEL_FILE_NAME.format(child_id=child_id)
        fog_model_file_path = os.path.join(CloudResourcesPaths.MODELS_FOLDER_PATH, fog_model_file_name)

        with open(fog_model_file_path, "wb") as fog_model_file:
            fog_model_file.write(fog_model_file_bytes)

        logger.info(f"Received message from fog {child_id}. The fog model was saved successfully.")

        CloudService.received_fog_messages[child_id] = {
            "fog_model_file_path": fog_model_file_path,
            "lambda_prev": lambda_prev_value
        }

        # Once messages from all fogs are received, process them
        if len(CloudService.received_fog_messages) == len(NodeState.get_current_node().child_nodes):
            logger.info("Received messages from all fogs. Processing received fog models:")
            for fog_id, fog_info in CloudService.received_fog_messages.items():
                logger.info(f"Processing fog model from {fog_id} located at {fog_info['fog_model_file_path']}")
                # Here you can load the model file, read lambda_prev, etc.
            logger.info("Stopping the cooling process.")
            CloudService.cloud_cooling_scheduler.stop_cooling()
            aggregate_fog_models(CloudService.received_fog_messages)
            # Delete the received fog model files (keeping only the cloud model)
            delete_all_files_in_folder(CloudResourcesPaths.MODELS_FOLDER_PATH, filter_string="fog")
            CloudService.cloud_service_state = CloudServiceState.IDLE

    @staticmethod
    def listen_to_receive_fog_queue():
        """Start listening to the RabbitMQ queue for incoming messages.
        This stops automatically when the cooling scheduler stops."""
        while True:
            connection = None
            try:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=CloudService.CLOUD_RABBITMQ_HOST,
                        heartbeat=30
                    )
                )
                channel = connection.channel()

                def callback(ch, _method, _properties, body):
                    message = json.loads(body.decode('utf-8'))

                    if int(message.get("scope")) == ModelScope.TRAINING.value:
                        CloudService.get_fog_model(message)
                        # check if cooling has stopped
                        if not CloudService.cloud_cooling_scheduler.is_cloud_cooling_operational():
                            logger.info("Stopping queue listener as cooling process is complete.")
                            ch.stop_consuming()
                    elif int(message.get("scope")) == ModelScope.EVALUATION.value:
                        CloudService.get_fog_evaluation_results(message)
                    else:
                        logger.warning(f"Received message from fog with scope {message.get('scope')} being not "
                                       f"recognized.")

                logger.info("Listening for messages from fog nodes...")
                channel.basic_consume(
                    queue=CloudService.FOG_CLOUD_RECEIVE_QUEUE,
                    on_message_callback=callback,
                    auto_ack=True
                )
                channel.start_consuming()
                # TODO: implement to recognize the evaluation performance coming from fogs (edges)
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.StreamLostError) as e:
                logger.error(f"Connection error in listen_to_receive_fog_queue: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in listen_to_receive_fog_queue: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception as e:
                        logger.error(f"An error occurred in listen_to_receive_fog_queue finally clause: {e}")

    @staticmethod
    def perform_model_evaluation(evaluation_date: str):
        CloudService.cloud_service_state = CloudServiceState.OPERATIONAL
        CloudService.evaluation_received_results_counter = 0

        if CloudService.CLOUD_RABBITMQ_HOST == "cloud-rabbitmq-host":
            raise ValueError("RabbitMQ host is not initialized. Please wait for the node to be detected.")

        cloud_model_file_path = os.path.join(CloudResourcesPaths.MODELS_FOLDER_PATH,
                                             CloudResourcesPaths.CLOUD_MODEL_FILE_NAME)

        cache_cloud_model_file_path = os.path.join(SharedResourcesPaths.CACHE_FOLDER_PATH,
                                                   CloudResourcesPaths.CLOUD_MODEL_FILE_NAME)

        if os.path.exists(cloud_model_file_path):
            logger.info("Cloud model exists in the model folder.")
            with open(cloud_model_file_path, "rb") as model_file:
                model_bytes = model_file.read()
                encoded_model = base64.b64encode(model_bytes).decode('utf-8')
        elif os.path.exists(cache_cloud_model_file_path):
            with open(cache_cloud_model_file_path, "rb") as model_file:
                model_bytes = model_file.read()
                encoded_model = base64.b64encode(model_bytes).decode('utf-8')
        else:
            return {"error": "No cloud model found, neither in container, neither in volume."}

        payload = {
            "start_date": None,
            "current_date": evaluation_date,
            "model_file": encoded_model,
            "scope": ModelScope.EVALUATION.value
        }

        CloudService.start_listener()
        CloudService._send_message_to_children(payload)

    @staticmethod
    def get_fog_evaluation_results(message):
        logger.info(f"Processing evaluation result received from fog: {message}")
        fog_id = message.get("fog_id")
        results = message.get("results", [])
        evaluation_date = message.get("evaluation_date")

        # Create a new record for this fog's evaluation and append it to the list.
        record = {
            "fog_id": fog_id,
            "results": results,
            "evaluation_date": evaluation_date
        }
        CloudService.received_fog_results.append(record)
        CloudService.evaluation_received_results_counter += 1

        # When results from all fog nodes have been received, finalize and save the performance data.
        if CloudService.evaluation_received_results_counter == len(NodeState.get_current_node().child_nodes):
            CloudService.cloud_service_state = CloudServiceState.IDLE
            CloudService.save_model_performance_to_json()

    @staticmethod
    def load_model_performance_from_json() -> list:
        """
        Load the model performance results from the JSON file.
        Returns a list of performance records (each a dict) if the file exists, or an empty list.
        """
        model_performance_file_path = os.path.join(
            SharedResourcesPaths.CACHE_FOLDER_PATH,
            CloudResourcesPaths.MODEL_PERFORMANCE_RESULTS_FILE_NAME
        )
        if os.path.exists(model_performance_file_path):
            try:
                with open(model_performance_file_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"Loaded {len(data)} existing performance records.")
                        return data
                    else:
                        logger.warning("Performance file format unexpected; starting with empty list.")
                        return []
            except Exception as e:
                logger.error(f"Error loading model performance file: {e}")
                return []
        else:
            return []

    @staticmethod
    def save_model_performance_to_json() -> None:
        """
        Append the new evaluation records from CloudService.received_fog_results to the existing
        performance data stored in the JSON file.
        """
        model_performance_file_path = os.path.join(
            SharedResourcesPaths.CACHE_FOLDER_PATH,
            CloudResourcesPaths.MODEL_PERFORMANCE_RESULTS_FILE_NAME
        )
        # Load any existing data (a list of records)
        existing_data = CloudService.load_model_performance_from_json()
        # Assume that for evaluation we store a list of records.
        # CloudService.received_fog_results should be a list of new records for the current evaluation round.
        if not isinstance(CloudService.received_fog_results, list):
            logger.error("CloudService.received_fog_results is not a list; cannot save performance results.")
            return
        # Combine the old and new records
        combined_data = existing_data + CloudService.received_fog_results
        try:
            with open(model_performance_file_path, "w") as f:
                json.dump(combined_data, f, indent=2)
            logger.info(f"Saved {len(CloudService.received_fog_results)} new performance records, "
                        f"total now {len(combined_data)}.")
        except Exception as e:
            logger.error(f"Error saving model performance file: {e}")

    @staticmethod
    def get_model_performance_evaluation() -> dict:
        """
        Loads the performance evaluation records from the JSON file and returns them.
        """
        performance_records = CloudService.load_model_performance_from_json()
        return {"performance_results": performance_records}
