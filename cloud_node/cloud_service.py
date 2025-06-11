import asyncio
import base64
import json
import math
import os
import time
import threading
from enum import Enum
from typing import Dict, Any

import pika
from pika.exceptions import AMQPConnectionError
from shared.fed_node.node_state import NodeState
from shared.fed_node.fed_node import MessageScope
from shared.shared_resources_paths import SharedResourcesPaths
from cloud_node.model_manager import create_model, aggregate_fog_models
from cloud_node.cloud_resources_paths import CloudResourcesPaths
from cloud_node.cloud_cooling_scheduler import CloudCoolingScheduler
from shared.utils import delete_all_files_in_folder
from shared.monitoring_thread import MonitoringThread
from shared.logging_config import logger
from cloud_node.db_manager import (save_genetic_results_to_db, save_performance_results_to_db,
                                   save_prediction_results_to_db, load_prediction_results_from_db,
                                   load_performance_results_from_db, load_genetic_results_from_db, init_db,
                                   get_edge_node_by_device_mac, save_system_metrics_to_db, load_system_metrics_from_db)


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
    received_fog_performance_results_metrics = []
    received_fog_performance_results_predictions = []
    received_fog_performance_genetic_results = []
    received_fog_evolution_system_metrics = []
    evaluation_received_results_counter = 0
    rabbitmq_init_monitor_thread = None
    fog_models_listener_thread = None

    is_cache_active = False
    genetic_strategy = None
    model_type = None
    start_date = None
    current_date = None
    current_working_date = None

    cloud_service_state = CloudServiceState.IDLE
    federated_simulation_state = FederatedSimulationState.IDLE

    buffer = None
    websocket_connection = None
    websocket_loop = None

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
                    init_db()
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
            try:
                # Only join if the listener thread is not the current thread.
                if CloudService.fog_models_listener_thread != threading.current_thread():
                    CloudService.fog_models_listener_thread.join(timeout=5)
                else:
                    logger.warning("Attempted to join the current thread; skipping join.")
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
            cloud_model = create_model(model_type)
            cloud_model.save(cloud_model_file_path)

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
            "scope": MessageScope.TRAINING.value
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
                CloudService.send_status_update("success", f"Finished aggregating cloud model for date "
                                                           f"{CloudService.current_working_date}.")
                CloudService.get_fog_genetic_results()
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
            CloudService.send_status_update("success", f"Finished aggregating cloud model for date "
                                                       f"{CloudService.current_working_date}.")
            logger.info("Finished aggregating cloud model.")
            CloudService.get_fog_genetic_results()

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
                    # logger.info(f"Received message from fog: {message}")
                    if int(message.get("scope")) == MessageScope.TRAINING.value:
                        CloudService.get_fog_model(message)
                        # check if cooling has stopped
                        if not CloudService.cloud_cooling_scheduler.is_cloud_cooling_operational():
                            logger.info("Stopping queue listener as cooling process is complete.")
                            ch.stop_consuming()
                    elif int(message.get("scope")) == MessageScope.EVALUATION.value:
                        CloudService.get_fog_evaluation_results(message)
                    elif int(message.get("scope")) == MessageScope.TEST_DATA_ENOUGH_EXISTS.value:
                        CloudService.get_fog_test_result(message)
                    elif int(message.get("scope")) == MessageScope.GENETIC_LOGBOOK.value:
                        CloudService.get_fog_genetic_result(message)
                    elif int(message.get("scope")) == MessageScope.EVOLUTION_SYSTEM_METRICS.value:
                        CloudService.get_received_fog_evolution_system_metrics(message)
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
            "scope": MessageScope.EVALUATION.value
        }

        CloudService.start_listener()
        CloudService._send_message_to_children(payload)

    @staticmethod
    def get_fog_evaluation_results(message):
        logger.info(f"Processing evaluation result received from fog: fog_id: {message.get('fog_id')}.")
        fog_id = message.get("fog_id")
        fog_mac = message.get("fog_mac")
        results = message.get("results", [])
        evaluation_date = message.get("evaluation_date")

        # Separate out metrics and prediction pairs from the received results.
        metrics_records = []
        predictions_records = []
        for res in results:
            # If the result contains numeric metrics, save that record.
            if "metrics" in res:
                metrics_records.append({
                    #"edge_id": res.get("edge_id"),
                    "edge_mac": res.get("edge_mac"),
                    "metrics": res["metrics"],
                    "evaluation_date": evaluation_date
                })
            # If the result contains prediction pairs, save that record.
            if "prediction_pairs" in res:
                predictions_records.append({
                   # "edge_id": res.get("edge_id"),
                    "edge_mac": res.get("edge_mac"),
                    "prediction_pairs": res["prediction_pairs"],
                    "evaluation_date": evaluation_date
                })

        CloudService.received_fog_performance_results_metrics.append({
            #"fog_id": fog_id,
            "fog_mac": fog_mac,
            "results": metrics_records,
            "evaluation_date": evaluation_date
        })
        CloudService.received_fog_performance_results_predictions.append({
            #"fog_id": fog_id,
            "fog_mac": fog_mac,
            "results": predictions_records,
            "evaluation_date": evaluation_date
        })

        CloudService.evaluation_received_results_counter += 1

        # Once we've received results from all fog nodes...
        if CloudService.evaluation_received_results_counter == len(NodeState.get_current_node().child_nodes):
            CloudService.cloud_service_state = CloudServiceState.IDLE

            # Instead of saving JSON, use the DB manager to save the results.
            save_performance_results_to_db(CloudService.current_working_date,
                                           CloudService.received_fog_performance_results_metrics)
            save_prediction_results_to_db(CloudService.current_working_date,
                                          CloudService.received_fog_performance_results_predictions)
            CloudService.send_status_update("success", f"Finished evaluating cloud model for date "
                                                       f"{CloudService.current_working_date}.", 1)

    @staticmethod
    def get_model_performance_evaluation(data: dict) -> dict:

        metric = data.get("metric")
        metric_type = int(data.get("metric_type"))

        # --- Case 1: Prediction Metrics ---
        if metric_type == 2:
            prediction_results = load_prediction_results_from_db()

            edge_id = data.get("edge_id")
            edge_mac = data.get("edge_mac")
            if not edge_mac:
                return {"error": "No edge_mac provided for prediction metric."}

            edge_mac_str = str(edge_mac)
            filtered_results = {}
            if isinstance(prediction_results, dict):
                for eval_date, group in prediction_results.items():
                    for record in group.get("prediction_results", []):
                        if str(record.get("edge_mac")) == edge_mac_str:
                            current_date = record.get("evaluation_date", eval_date)
                            filtered_results.setdefault(current_date, []).extend(record.get("prediction_pairs", []))
            else:
                return {"error": "Unexpected data structure for prediction results."}

            daily_results = [
                {"evaluation_date": date, "prediction_pairs": filtered_results[date]}
                for date in sorted(filtered_results.keys())
            ]
            result = {"prediction_results": daily_results}
            return result

        # --- Case 2: Genetic Metrics ---
        elif metric_type == 3:
            genetic_results = load_genetic_results_from_db()
            # filter_fog_id = data.get("fog_id")
            filter_fog_mac = data.get("fog_mac")
            aggregated_genetic = {}

            if isinstance(genetic_results, dict):
                for eval_date, group in genetic_results.items():
                    for record in group.get("genetic_results", []):
                        if filter_fog_mac and str(record.get("fog_mac")) != str(filter_fog_mac):
                            continue
                        current_date = record.get("evaluation_date", eval_date)
                        if not current_date:
                            continue
                        aggregated_genetic.setdefault(current_date, {})
                        for rec in record.get("records", []):
                            gen_number = rec.get("gen")
                            if metric in rec and isinstance(rec[metric], (int, float)) and math.isfinite(rec[metric]):
                                aggregated_genetic[current_date][gen_number] = rec[metric]

            elif isinstance(genetic_results, list):
                for record in genetic_results:
                    if filter_fog_mac and str(record.get("fog_mac")) != str(filter_fog_mac):
                        continue
                    eval_date = record.get("evaluation_date")
                    if not eval_date:
                        continue
                    aggregated_genetic.setdefault(eval_date, {})
                    for rec in record.get("records", []):
                        gen_number = rec.get("gen")
                        if metric in rec and isinstance(rec[metric], (int, float)) and math.isfinite(rec[metric]):
                            aggregated_genetic[eval_date][gen_number] = rec[metric]
            else:
                return {"error": "Genetic results have an unrecognized structure."}

            # Build output for every evaluation date.
            aggregated_list = []
            for date in sorted(aggregated_genetic.keys()):
                generations = [
                    {"gen": gen, "value": aggregated_genetic[date][gen]}
                    for gen in sorted(aggregated_genetic[date].keys())
                ]
                aggregated_list.append({"evaluation_date": date, "generations": generations})
            result = {"genetic_results": aggregated_list, "selected_metric": metric}
            return result

        # --- Case 3: Numeric (Model Performance) Metrics ---
        elif metric_type == 1:
            performance_results = load_performance_results_from_db()
            from collections import defaultdict
            grouped_metrics = defaultdict(list)

            if isinstance(performance_results, dict):
                for eval_date, rec_list in performance_results.items():
                    for record in rec_list.get("performance_results", []):
                        for res in record.get("results", []):
                            metrics = res.get("metrics", {})
                            if metric in metrics and isinstance(metrics[metric], (int, float)):
                                grouped_metrics[eval_date].append(metrics[metric])
            else:
                return {"error": "Performance results have an unrecognized structure."}

            daily_averages = []
            for eval_date in sorted(grouped_metrics.keys()):
                values = grouped_metrics[eval_date]
                avg_value = sum(values) / len(values) if values else None
                daily_averages.append({"evaluation_date": eval_date, "average": avg_value})
            result = {"performance_results": daily_averages}
            return result

        # --- Case 4: System Metric -------
        else:
            system_metrics = load_system_metrics_from_db()
            # filter_fog_id = data.get("fog_id")
            filter_fog_mac = data.get("fog_mac")
            aggregated_system_metrics = {}

            if isinstance(system_metrics, dict):
                for eval_data, group in system_metrics.items():
                    for record in group.get("system_metrics", []):
                        if filter_fog_mac and str(record.get("fog_mac")) != str(filter_fog_mac):
                            continue
                        current_date = record.get("evaluation_date")
                        if not current_date:
                            continue
                        aggregated_system_metrics.setdefault(current_date, {})
                        for rec in record.get("system_metrics", []):
                            gen_number = rec.get("gen")
                            if metric in rec and isinstance(rec[metric], (int, float)) and math.isfinite(rec[metric]):
                                aggregated_system_metrics[current_date][gen_number] = rec[metric]
            elif isinstance(system_metrics, list):
                for record in system_metrics:
                    if filter_fog_mac and str(record.get("fog_mac")) != str(filter_fog_mac):
                        continue
                    eval_date = record.get("evaluation_date")
                    if not eval_date:
                        continue
                    aggregated_system_metrics.setdefault(eval_date, {})
                    for rec in record.get("system_metrics", []):
                        gen_number = rec.get("gen")
                        if metric in rec and isinstance(rec[metric], (int, float)) and math.isfinite(rec[metric]):
                            aggregated_system_metrics[eval_date][gen_number] = rec[metric]
            else:
                return {"error": "System metrics have an unrecognized structure."}

            aggregated_list = []

            for date in sorted(aggregated_system_metrics.keys()):
                generations = [
                    {"gen": gen, "value": aggregated_system_metrics[date][gen]}
                    for gen in sorted(aggregated_system_metrics[date].keys())
                ]
                aggregated_list.append({"evaluation_date": date, "generations": generations})
            result = {"system_metrics": aggregated_list, "selected_metric": metric}
            return result

    @staticmethod
    def get_available_performance_metrics() -> Dict[str, Any]:
        """
        Returns available performance metrics based on data stored in the database.
        Uses the DB load functions and returns a dictionary with three keys:
          - "model_performance_metrics"
          - "prediction_metrics"
          - "genetic_metrics"
        """
        # Load data from the database.
        performance_results = load_performance_results_from_db()
        prediction_results = load_prediction_results_from_db()
        genetic_results = load_genetic_results_from_db()
        system_metric_results = load_system_metrics_from_db()
        logger.info(f"System metrics: {system_metric_results}")

        # Gather numeric performance metrics.
        model_performance_metrics = set()
        # performance_results is a dict keyed by evaluation_date.
        for date, group in performance_results.items():
            # Each group should contain a "performance_results" key with a list of records.
            for record in group.get("performance_results", []):
                # Each record contains "results": a list of entries.
                for res in record.get("results", []):
                    metrics = res.get("metrics", {})
                    model_performance_metrics.update(metrics.keys())

        # Gather prediction metrics.
        # In our DB schema, prediction results always store a "prediction_pairs" key.
        prediction_metrics = set()
        for date, group in prediction_results.items():
            for record in group.get("prediction_results", []):
                if record.get("prediction_pairs") and len(record.get("prediction_pairs")) > 0:
                    prediction_metrics.add("prediction_pairs")

        # Gather genetic metrics.
        # "nevals", "avg", "std", "min", "max", "genotypic_diversity", "phenotypic_diversity"
        genetic_metrics = set()
        for date, group in genetic_results.items():
            for record in group.get("genetic_results", []):
                # Each record should have a "records" key with a list of generation dictionaries.
                for rec in record.get("records", []):
                    for key in rec.keys():
                        if key != "gen":  # We consider "gen" as an index; other keys are metric names.
                            genetic_metrics.add(key)

        system_metric_metrics = set()
        for date, group in system_metric_results.items():
            for record in group.get("system_metrics", []):
                for rec in record.get("system_metrics", []):
                    for key in rec.keys():
                        if key != "gen":
                            system_metric_metrics.add(key)

        return {
            "model_performance_metrics": sorted(list(model_performance_metrics)),
            "prediction_metrics": sorted(list(prediction_metrics)),
            "genetic_metrics": sorted(list(genetic_metrics)),
            "system_metric_metrics": sorted(list(system_metric_metrics)),
        }

    @staticmethod
    def check_enough_data_existence(data, scope: MessageScope):
        CloudService.cloud_service_state = CloudServiceState.OPERATIONAL
        CloudService.evaluation_received_results_counter = 0
        CloudService.start_listener()

        CloudService.current_working_date = data.get("end_date")

        if "start_date" in data:
            payload = {
                "scope": MessageScope.TEST_DATA_ENOUGH_EXISTS.value,
                "start_date": data.get("start_date"),
                "current_date": data.get("end_date"),
            }
        else:
            payload = {
                "scope": MessageScope.TEST_DATA_ENOUGH_EXISTS.value,
                "current_date": data.get("end_date"),
                "is_training_validation": scope.value == MessageScope.TRAINING.value
            }

        logger.info(f"Message received before checking existance: {data}")
        CloudService.buffer = data
        CloudService.buffer["scope"] = scope.value
        CloudService._send_message_to_children(payload)

    @staticmethod
    def get_fog_test_result(message):
        if message.get("enough_data_existence"):
            CloudService.evaluation_received_results_counter += 1
        else:
            CloudService.send_status_update("error", f"Found insufficient data on edge. Please skip day "
                                                     f"{CloudService.current_working_date}.")
            CloudService.cloud_service_state = CloudServiceState.IDLE
            return

        if CloudService.evaluation_received_results_counter == len(NodeState.get_current_node().child_nodes):
            if not CloudService.buffer:
                logger.error("Buffer data is not set. Cannot proceed with training process execution.")
                CloudService.send_status_update("error", "Buffer data is missing; please retry operation.")
                CloudService.cloud_service_state = CloudServiceState.IDLE
                return

            CloudService.send_status_update("success", f"Found sufficient data on edge for date "
                                                       f"{CloudService.current_working_date}.")

            CloudService.received_fog_performance_results_metrics = []
            CloudService.received_fog_performance_results_predictions = []
            CloudService.received_fog_performance_genetic_results = []
            CloudService.received_fog_evolution_system_metrics = []

            if CloudService.buffer.get("scope") == MessageScope.TRAINING.value:
                if "start_date" in CloudService.buffer:
                    CloudService.execute_training_process(
                        CloudService.buffer.get("start_date"),
                        CloudService.buffer.get("end_date"),
                        CloudService.buffer.get("is_cache_active"),
                        CloudService.buffer.get("genetic_evaluation_strategy"),
                        CloudService.buffer.get("model_type")
                    )
                else:
                    CloudService.execute_training_process(
                        None,
                        CloudService.buffer.get("end_date"),
                        CloudService.buffer.get("is_cache_active"),
                        CloudService.buffer.get("genetic_evaluation_strategy"),
                        CloudService.buffer.get("model_type")
                    )
            elif CloudService.buffer.get("scope") == MessageScope.EVALUATION.value:
                CloudService.perform_model_evaluation(CloudService.buffer.get("end_date"))

            CloudService.send_status_update("success", f"Processing complete for date "
                                                       f"{CloudService.current_working_date}.")

    @staticmethod
    def send_status_update(status: str, message: str, code: int = 0):
        """
        Sends a status update to the frontend.
        status: one of "error", "warning", "success"
        message: The status message string.
        """
        update_payload = {
            "type": "status_update",
            "status": status,
            "message": message,
            "timestamp": time.time(),
            "code": code
        }

        if CloudService.websocket_connection and CloudService.websocket_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    CloudService.websocket_connection.send_json(update_payload),
                    CloudService.websocket_loop
                )
                logger.info(f"Sent status update: {update_payload}")
            except Exception as e:
                logger.error(f"Error sending status update: {e}")
        else:
            logger.warning("No websocket connection available to send status update.")

    @staticmethod
    def get_fog_genetic_results():
        logger.info("Sending request for genetic results to fogs.")
        payload = {
            "scope": MessageScope.GENETIC_LOGBOOK.value
        }
        CloudService._send_message_to_children(payload)

    @staticmethod
    def get_fog_evolution_system_metrics():
        logger.info("Sending request for evolution system metrics to fogs.")
        payload = {
            "scope": MessageScope.EVOLUTION_SYSTEM_METRICS.value
        }
        CloudService._send_message_to_children(payload)

    @staticmethod
    def get_fog_genetic_result(message):
        CloudService.received_fog_performance_genetic_results.append(message)

        if len(CloudService.received_fog_performance_genetic_results) == len(NodeState.get_current_node().child_nodes):
            save_genetic_results_to_db(CloudService.current_working_date,
                                       CloudService.received_fog_performance_genetic_results)
            CloudService.get_fog_evolution_system_metrics()

    @staticmethod
    def get_received_fog_evolution_system_metrics(message):
        CloudService.received_fog_evolution_system_metrics.append(message)

        if len(CloudService.received_fog_evolution_system_metrics) == len(NodeState.get_current_node().child_nodes):
            save_system_metrics_to_db(CloudService.current_working_date,
                                      CloudService.received_fog_evolution_system_metrics)

    @staticmethod
    def get_cloud_temperature():
        if CloudService.cloud_cooling_scheduler.is_cloud_cooling_operational():
            return {"cloud_temperature": CloudService.cloud_cooling_scheduler.temperature}
        return {"cloud_temperature": 0.0}

    @staticmethod
    def get_edge_node_record_from_cloud_db(device_mac: str):
        return get_edge_node_by_device_mac(device_mac)
