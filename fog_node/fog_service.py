# fog_node/fog_service.py

import base64
import json
import os
import threading
import time
from enum import Enum
from typing import Dict, Any

import pika
from pika.exceptions import AMQPConnectionError
from shared.fed_node.node_state import NodeState
from fog_node.fog_resources_paths import FogResourcesPaths
from fog_node.genetics.genetic_engine import GeneticEngine
from fog_node.fog_cooling_scheduler import FogCoolingScheduler
from fog_node.model_manager import execute_models_aggregation, read_lambda_prev
from shared.monitoring_thread import MonitoringThread
from shared.shared_resources_paths import SharedResourcesPaths
from shared.utils import delete_all_files_in_folder
from shared.logging_config import logger
from shared.fed_node.fed_node import MessageScope


class FogServiceState(Enum):
    IDLE = 1
    GENETIC_EVALUATION = 2
    TRAINING = 3
    AGGREGATION = 4


class FogService:
    CLOUD_RABBITMQ_HOST = "cloud-rabbitmq-host"
    CLOUD_FOG_RECEIVE_EXCHANGE = "cloud_to_fog_exchange"
    CLOUD_FOG_RECEIVE_QUEUE = "cloud_to_fog_"
    FOG_CLOUD_SEND_QUEUE = "fog_to_cloud_queue"

    FOG_RABBITMQ_HOST = "fog-rabbitmq-host"
    FOG_EDGE_SEND_EXCHANGE = "fog_to_edge_exchange"
    EDGE_FOG_RECEIVE_QUEUE = "edge_to_fog_queue"

    genetic_engine = GeneticEngine()
    fog_cooling_scheduler: FogCoolingScheduler = None
    process_init_monitor_thread = None
    fog_service_state = FogServiceState.IDLE

    edge_evaluation_performances = {}
    edge_responses_counter = 0
    messages_to_trainable_edges = {}

    @staticmethod
    def get_fog_service_state():
        return {
            "fog_service_state": FogService.fog_service_state.value
        }

    @classmethod
    def get_fog_cooling_scheduler(cls):
        if cls.fog_cooling_scheduler is None:
            cls.fog_cooling_scheduler = FogCoolingScheduler(target=cls.send_fog_model_to_cloud)
        return cls.fog_cooling_scheduler

    @staticmethod
    def set_genetic_parameters(population_size, number_of_generations, stagnation_limit):
        # Pass the values to the genetic engine's setter.
        FogService.genetic_engine.set_genetic_engine_parameters(
            population_size, number_of_generations, stagnation_limit
        )
        logger.info(f"Genetic engine parameters updated: population_size={population_size}, "
                    f"number_of_generations={number_of_generations}, stagnation_limit={stagnation_limit}")

    @staticmethod
    def get_genetic_engine_parameters():
        return FogService.genetic_engine.get_genetic_engine_parameters()

    @staticmethod
    def set_training_parameters(lr_min, lr_max, bs_min, bs_max, ep_min, ep_max, pa_min, pa_max, ftl_min, ftl_max):
        FogService.genetic_engine.configure_training_parameters_bounds(lr_min, lr_max, bs_min, bs_max, ep_min, ep_max,
                                                                       pa_min, pa_max, ftl_min, ftl_max)

    @staticmethod
    def get_training_parameters():
        return FogService.genetic_engine.get_current_training_parameter_bounds()

    @staticmethod
    def monitor_parent_children_nodes_and_init_process() -> None:
        """
        Monitoring the parent and children nodes of the current fog node and initialize process when they are set.
        """

        current_node = NodeState.get_current_node()
        if current_node and current_node.parent_node and current_node.child_nodes:
            logger.info(f"Parent node detected: {current_node.parent_node}.")
            FogService.CLOUD_RABBITMQ_HOST = current_node.parent_node.ip_address
            FogService.FOG_RABBITMQ_HOST = current_node.ip_address
            for child_node in current_node.child_nodes:
                logger.info(f"Child node detected: {child_node}.")

            FogService.init_process()
            if FogService.process_init_monitor_thread:
                FogService.process_init_monitor_thread.stop()

    @staticmethod
    def start_monitoring_parent_children_nodes() -> None:
        """
        Start monitoring thread to check for the parent node and children nodes and initialize the process then.
        """
        FogService.process_init_monitor_thread = MonitoringThread(
            target=FogService.monitor_parent_children_nodes_and_init_process,
            sleep_time=2
        )
        FogService.process_init_monitor_thread.start()

    @staticmethod
    def init_process() -> None:
        FogService.init_rabbitmq()
        FogService.genetic_engine.setup(FogService.FOG_RABBITMQ_HOST, FogService.FOG_EDGE_SEND_EXCHANGE,
                                        FogService.EDGE_FOG_RECEIVE_QUEUE)
        FogService.genetic_engine.set_cloud_node(NodeState.get_current_node().parent_node)
        cloud_listener = threading.Thread(target=FogService.listen_to_cloud_receiving_queue, daemon=True)
        cloud_listener.start()

        edge_listener = threading.Thread(target=FogService.listen_to_edges_receiving_queue, daemon=True)
        edge_listener.start()

    @staticmethod
    def init_rabbitmq() -> None:
        """
        Initialize RabbitMQ connection and declare queues.
        """
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=FogService.CLOUD_RABBITMQ_HOST))
        channel = connection.channel()

        # declare the receiving exchange and queue for listening and sending messages from/to the cloud
        channel.exchange_declare(exchange=FogService.CLOUD_FOG_RECEIVE_EXCHANGE, exchange_type="direct", durable=True)
        fog_id = str(NodeState.get_current_node().id)
        FogService.CLOUD_FOG_RECEIVE_QUEUE += str(fog_id)
        channel.queue_declare(queue=FogService.CLOUD_FOG_RECEIVE_QUEUE, durable=True)
        channel.queue_bind(queue=FogService.CLOUD_FOG_RECEIVE_QUEUE, exchange=FogService.CLOUD_FOG_RECEIVE_EXCHANGE,
                           routing_key=fog_id)

        channel.queue_declare(queue=FogService.FOG_CLOUD_SEND_QUEUE, durable=True)
        connection.close()

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=FogService.FOG_RABBITMQ_HOST))
        channel = connection.channel()

        # declare the receiving exchange for listening and sending messages from/to the edge
        channel.exchange_declare(exchange=FogService.FOG_EDGE_SEND_EXCHANGE, exchange_type="direct", durable=True)
        channel.queue_declare(queue=FogService.EDGE_FOG_RECEIVE_QUEUE, durable=True)
        connection.close()

    @staticmethod
    def listen_to_cloud_receiving_queue() -> None:
        """
        Start listening to the RabbitMQ queue for messages from the cloud node.
        """
        while True:
            connection = None
            try:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=FogService.CLOUD_RABBITMQ_HOST,
                        heartbeat=30
                    )
                )
                channel = connection.channel()

                def callback(ch, method, _properties, body):
                    """
                    RabbitMQ callback to process messages received from the cloud queue.
                    """
                    try:
                        # Acknowledge the message immediately to remove it from the queue.
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        # Then process the message.
                        message = json.loads(body.decode("utf-8"))
                        child_id = message.get("child_id")
                        scope = message.get("scope")
                        logger.info(f"Processing message for child id {child_id}.")

                        scheduler = FogService.get_fog_cooling_scheduler()
                        if scheduler is not None:
                            scheduler.reset()
                        else:
                            logger.error("Fog cooling scheduler is not initialized.")

                        if scope == MessageScope.TRAINING.value:
                            FogService.orchestrate_training(message)
                        elif scope == MessageScope.EVALUATION.value:
                            FogService.edge_evaluation_performances = {}
                            FogService.orchestrate_evaluation(message)
                        elif scope == MessageScope.TEST_DATA_ENOUGH_EXISTS.value:
                            FogService.edge_responses_counter = 0
                            FogService.orchestrate_enough_data_testing(message)
                        elif scope == MessageScope.GENETIC_LOGBOOK.value:
                            FogService.send_fog_model_to_cloud(MessageScope.GENETIC_LOGBOOK, None)
                        elif scope == MessageScope.EVOLUTION_SYSTEM_METRICS.value:
                            FogService.send_fog_model_to_cloud(MessageScope.EVOLUTION_SYSTEM_METRICS, None)
                        else:
                            logger.warning(f"There was met a an unknown scope of the model {scope}.")
                    except Exception as e1:
                        logger.error(f"Error processing message: {e1}")
                        # in case of error we must negatively acknowledge the message and requeue it
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

                channel.basic_consume(queue=FogService.CLOUD_FOG_RECEIVE_QUEUE, on_message_callback=callback,
                                      auto_ack=False)
                logger.info("Listening for messages from the cloud...")
                channel.start_consuming()
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.StreamLostError) as e:
                logger.error(f"Connection error in listen_to_cloud_receiving_queue: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in listen_to_cloud_receiving_queue: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception as e:
                        logger.error(f"An error occurred in listen_to_cloud_receiving_queue finally clause: {e}")

    @staticmethod
    def orchestrate_training(message):
        """
        Process a received message from cloud node.
        :param message: The received message containing the payload and model file.
        """
        try:
            start_date = message.get("start_date")
            current_date = message.get("end_date")
            is_cache_active = message.get("is_cache_active")
            genetic_evaluation_strategy = message.get("genetic_evaluation_strategy")
            model_type = message.get("model_type")
            model_file_base64 = message.get("model_file")

            if not model_file_base64:
                raise ValueError("Model file is missing in the received message.")

            # decode the base64 model file
            model_file_bytes = base64.b64decode(model_file_base64)

            # save the model file locally
            fog_model_file_path = os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH,
                                               FogResourcesPaths.FOG_MODEL_FILE_NAME)

            os.makedirs(FogResourcesPaths.MODELS_FOLDER_PATH, exist_ok=True)  # ensure the directory exists
            with open(fog_model_file_path, "wb") as model_file:
                model_file.write(model_file_bytes)

            logger.info(f"Received cloud model saved at: {fog_model_file_path}")

            # handle the received payload
            logger.info(f"Processing with start_date: {start_date}, current_date: {current_date}, cache_active: "
                        f"{is_cache_active}, strategy: {genetic_evaluation_strategy}")

            # run the genetic engine
            if start_date is not None:
                FogService.genetic_engine.set_operating_data_date([start_date, current_date])
                # to not inherit population from a previous simulation
                delete_all_files_in_folder(SharedResourcesPaths.CACHE_FOLDER_PATH, "genetic")
            else:
                FogService.genetic_engine.set_operating_data_date([current_date])
            FogService.fog_service_state = FogServiceState.GENETIC_EVALUATION
            FogService.genetic_engine.evolve()
            # get top individuals for each fog child
            top_individuals = FogService.genetic_engine.get_top_k_individuals(len(NodeState.get_current_node()
                                                                                  .child_nodes))
            FogService.fog_service_state = FogServiceState.TRAINING
            FogService.forward_model_to_edges(model_file_base64, start_date, current_date, is_cache_active,
                                              genetic_evaluation_strategy, model_type, top_individuals)
            FogService.genetic_engine.save_population_to_json()

        except Exception as e:
            logger.error(f"Error processing the received message: {str(e)}")

    @staticmethod
    def forward_model_to_edges(model_file_base64: str, start_date: str, current_date: str, is_cache_active: bool,
                               genetic_evaluation_strategy: str, model_type: str, top_individuals) -> None:
        """
        Forward the received model and payload to all child edge nodes.
        """
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=FogService.FOG_RABBITMQ_HOST,
                heartbeat=30
            )
        )
        channel = connection.channel()

        current_node = NodeState.get_current_node()
        if not current_node:
            raise ValueError("Current fog node is not initialized.")

        trainable_edges = NodeState.get_current_node().child_nodes

        logger.info(f"Forwarding the model and parameters to {len(trainable_edges)}.")
        logger.info(top_individuals)

        FogService.messages_to_trainable_edges = {}

        for index, edge_node in enumerate(trainable_edges):
            individual = top_individuals[index]
            message = {
                "child_id": edge_node.id,
                "scope": MessageScope.TRAINING.value,
                "genetic_evaluation": False,
                "start_date": start_date,
                "current_date": current_date,
                "is_cache_active": is_cache_active,
                "genetic_evaluation_strategy": genetic_evaluation_strategy,
                "model_type": model_type,
                "model_file": model_file_base64,
                "learning_rate": individual[0] / 10000.0,
                "batch_size": individual[1],
                "epochs": individual[2],
                "patience": individual[3],
                "fine_tune_layers": individual[4]
            }
            FogService.messages_to_trainable_edges[edge_node.id] = message
            routing_key = str(edge_node.id)
            channel.basic_publish(exchange=FogService.FOG_EDGE_SEND_EXCHANGE, routing_key=routing_key,
                                  body=json.dumps(message))
            logger.info(f"Forwarded model to edge {edge_node.name} ({edge_node.ip_address}:{edge_node.port})")

        connection.close()
        scheduler = FogService.get_fog_cooling_scheduler()
        scheduler.start_cooling()

    @staticmethod
    def listen_to_edges_receiving_queue() -> None:
        """
        Start listening to RabbitMQ queue for messages from the edges node with the trained models.
        """
        while True:
            connection = None
            try:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=FogService.FOG_RABBITMQ_HOST,
                        heartbeat=30
                    )
                )

                channel = connection.channel()

                def callback(_ch, _method, _properties, body):
                    """
                    RabbitMQ callback to process messages received from the edges queue
                    """
                    message = json.loads(body.decode('utf-8'))

                    if message.get("scope") == MessageScope.TRAINING.value:
                        FogService.fog_cooling_scheduler.increment_stopping_score()
                        FogService.get_edge_model(message)
                        FogService.fog_cooling_scheduler.has_reached_stopping_condition_for_cooler()

                        if not FogService.fog_cooling_scheduler.is_fog_cooling_operational():
                            logger.info("Stopping queue listener as cooling process is complete.")
                            channel.stop_consuming()
                    elif message.get("scope") == MessageScope.EVALUATION.value:
                        FogService.edge_evaluation_performances[message.get("edge_id")] = (message.get("metrics"),
                                                                                           message
                                                                                           .get("prediction_pairs"))

                        children_number = len(NodeState.get_current_node().child_nodes)
                        if len(FogService.edge_evaluation_performances) == children_number:
                            logger.info("Stopping queue listener as receiving evaluation metrics process is complete.")
                            channel.stop_consuming()
                            params = {
                                "evaluation_date": message.get("evaluation_date"),
                            }
                            FogService.send_fog_model_to_cloud(MessageScope.EVALUATION, params)
                    elif message.get("scope") == MessageScope.TEST_DATA_ENOUGH_EXISTS.value:
                        if message.get("enough_data_existence"):
                            FogService.edge_responses_counter += 1
                        else:
                            logger.info("Stopping queue listener as receiving not enough data existence process is "
                                        "complete.")
                            channel.stop_consuming()
                            params = {
                                "enough_data_existence": False
                            }
                            FogService.send_fog_model_to_cloud(MessageScope.TEST_DATA_ENOUGH_EXISTS, params)

                        children_number = len(NodeState.get_current_node().child_nodes)
                        if FogService.edge_responses_counter == children_number:
                            logger.info("Stopping queue listener as receiving enough data existence process is "
                                        "complete.")
                            channel.stop_consuming()
                            params = {
                                "enough_data_existence": True
                            }
                            FogService.send_fog_model_to_cloud(MessageScope.TEST_DATA_ENOUGH_EXISTS, params)

                logger.info("Listening for messages from edge nodes...")
                channel.basic_consume(queue=FogService.EDGE_FOG_RECEIVE_QUEUE, on_message_callback=callback,
                                      auto_ack=True)
                channel.start_consuming()
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.StreamLostError) as e:
                logger.error(f"Connection error in listen_to_edges_receiving_queue: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in listen_to_edges_receiving_queue: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception as e:
                        logger.error(f"An error occurred in listen_to_edges_receiving_queue finally clause: {e}")

    @staticmethod
    def get_edge_model(message):
        try:
            if not FogService.fog_cooling_scheduler.is_fog_cooling_operational():
                return
            edge_model_file_bytes = base64.b64decode(message["model_file"])
            metrics = message.get("metrics")
            edge_id = message.get("edge_id")

            edge_model_file_name = FogResourcesPaths.EDGE_MODEL_FILE_NAME
            edge_model_file_path = os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH, edge_model_file_name)

            with open(edge_model_file_path, "wb") as edge_model_file:
                edge_model_file.write(edge_model_file_bytes)

            logger.info(f"Received message from edge {edge_id}. Continuing with model aggregation.")

            FogService.fog_service_state = FogServiceState.AGGREGATION
            execute_models_aggregation(FogService.fog_cooling_scheduler, metrics)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

        # deleting the received edge model file keeping only the fog model
        delete_all_files_in_folder(FogResourcesPaths.MODELS_FOLDER_PATH, filter_string="edge")

    @staticmethod
    def send_fog_model_to_cloud(model_scope: MessageScope, params):
        message = None
        if model_scope.value == MessageScope.TRAINING.value:
            logger.info("Running the sending of the fog model to cloud.")

            fog_model_file_path = os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH,
                                               FogResourcesPaths.FOG_MODEL_FILE_NAME)
            with open(fog_model_file_path, "rb") as model_file:
                model_bytes = model_file.read()
                model_file_base64 = base64.b64encode(model_bytes).decode("utf-8")

            message = {
                "scope": model_scope.value,
                "fog_id": NodeState.get_current_node().id,
                "lambda_prev": read_lambda_prev(),
                "model_file": model_file_base64
            }
        elif model_scope.value == MessageScope.EVALUATION.value:
            logger.info("Running the sending of edge results to cloud.")

            results = []
            for edge_id, (metrics, prediction_pairs) in FogService.edge_evaluation_performances.items():
                results.append({
                    "edge_id": edge_id,
                    "metrics": metrics,
                    "prediction_pairs": prediction_pairs
                })
            message = {
                "scope": model_scope.value,
                "fog_id": NodeState.get_current_node().id,
                "results": results,
                "evaluation_date": params.get("evaluation_date")
            }
        elif model_scope.value == MessageScope.TEST_DATA_ENOUGH_EXISTS.value:
            logger.info("Running the sending of the edge enough data confirmation to cloud.")
            message = {
                "scope": MessageScope.TEST_DATA_ENOUGH_EXISTS.value,
                "enough_data_existence": params.get("enough_data_existence")
            }
        elif model_scope.value == MessageScope.GENETIC_LOGBOOK.value:
            logger.info("Running sending the genetic logbook to cloud.")
            message = FogService.get_genetic_logbook()

        elif model_scope.value == MessageScope.EVOLUTION_SYSTEM_METRICS.value:
            logger.info("Running sending the evolution system metrics to cloud.")
            message = FogService.get_evolution_system_metrics()

        published = False
        while not published:
            try:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=FogService.CLOUD_RABBITMQ_HOST,
                        heartbeat=30
                    )
                )
                channel = connection.channel()
                channel.basic_publish(exchange="",
                                      routing_key=FogService.FOG_CLOUD_SEND_QUEUE,
                                      body=json.dumps(message))
                connection.close()
                published = True

                if model_scope.value == MessageScope.TRAINING.value:
                    logger.info("Fog has sent fog model to cloud node.")
                else:
                    logger.info("Fog has sent edge results to cloud node.")
            except Exception as e:
                logger.error(f"Error sending fog model to cloud: {e}. Retrying in 5 seconds...")
                time.sleep(5)

        # Restart the evaluation/training node assignment
        for node in NodeState.get_current_node().child_nodes:
            node.is_evaluation_node = False

        FogService.fog_service_state = FogServiceState.IDLE

    @staticmethod
    def orchestrate_evaluation(message):
        try:
            logger.info("step1")
            model_file_base64 = message.get("model_file")
            logger.info("step2")
            if not model_file_base64:
                raise ValueError("Model file is missing in the received message.")

            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=FogService.FOG_RABBITMQ_HOST,
                    heartbeat=30
                )
            )
            channel = connection.channel()
            logger.info("step3")
            current_node = NodeState.get_current_node()
            if not current_node:
                raise ValueError("Current fog node is not initialized.")

            if FogService.genetic_engine.current_population is None:
                FogService.genetic_engine.load_population_from_json()
            logger.info("step4")
            # TODO: update such that if not enough individuals to add some random
            top_individuals = FogService.genetic_engine.get_top_k_individuals(len(NodeState.get_current_node()
                                                                                  .child_nodes))
            logger.info("step5")
            logger.info(top_individuals)

            for index, edge_node in enumerate(current_node.child_nodes):
                routing_key = str(edge_node.id)
                individual = top_individuals[index]
                message["learning_rate"] = individual[0] / 10000.0
                message["batch_size"] = individual[1]
                message["epochs"] = individual[2]
                message["patience"] = individual[3]
                message["fine_tune_layers"] = individual[4]

                channel.basic_publish(exchange=FogService.FOG_EDGE_SEND_EXCHANGE, routing_key=routing_key,
                                      body=json.dumps(message))
                logger.info(f"Forwarded evaluation model to edge {edge_node.name} ({edge_node.ip_address}:"
                            f"{edge_node.port})")

            connection.close()
            # TODO: implement on edge the receiving, evaluation and send back of the result
        except Exception as e:
            logger.warning(f"An error occurred during orchestration evaluation: {e}")

    @staticmethod
    def orchestrate_enough_data_testing(message):
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=FogService.FOG_RABBITMQ_HOST,
                    heartbeat=30
                )
            )
            channel = connection.channel()

            current_node = NodeState.get_current_node()
            if not current_node:
                raise ValueError("Current fog node is not initialized.")

            for index, edge_node in enumerate(current_node.child_nodes):
                routing_key = str(edge_node.id)
                channel.basic_publish(exchange=FogService.FOG_EDGE_SEND_EXCHANGE, routing_key=routing_key,
                                      body=json.dumps(message))
                logger.info(f"Forwarded enough data tes to edge {edge_node.name} ({edge_node.ip_address}:"
                            f"{edge_node.port})")

            connection.close()
        except Exception as e:
            logger.warning(f"An error occurred during orchestration evaluation: {e}")

    @staticmethod
    def get_genetic_logbook() -> Dict[str, Any]:
        """
        Returns all logbook records (i.e. statistics for each generation)
        from the current evolution iteration of the genetic engine as a JSON-compatible dictionary.
        If the logbook is empty, it attempts to load it from the saved file.
        """
        if not FogService.genetic_engine.logbook or len(FogService.genetic_engine.logbook) == 0:
            logger.info("Genetic logbook is empty, attempting to load from saved file...")
            FogService.genetic_engine.load_population_from_json()

        # Instead of returning just the latest record, return all records from the current evolution iteration.
        return {
            "header": FogService.genetic_engine.DEFAULT_LOGBOOK_HEADER,
            "records": FogService.genetic_engine.logbook,
            "fog_id": NodeState.get_current_node().id,
            "scope": MessageScope.GENETIC_LOGBOOK.value
        }

    @staticmethod
    def handle_edge_node_unfinished_previous_round(edge_id: str):
        trainable_edges = NodeState.get_current_node().child_nodes
        tricky_node = next((node for node in trainable_edges if node.id == edge_id), None)

        if tricky_node is not None:
            time.sleep(360)
            message = FogService.messages_to_trainable_edges[edge_id]

            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=FogService.FOG_RABBITMQ_HOST,
                    heartbeat=30
                )
            )
            channel = connection.channel()
            routing_key = str(edge_id)
            channel.basic_publish(exchange=FogService.FOG_EDGE_SEND_EXCHANGE, routing_key=routing_key,
                                  body=json.dumps(message))
            logger.info(f"Forwarded model to edge {tricky_node.name} ({tricky_node.ip_address}:"
                        f"{tricky_node.port})")
            connection.close()
        else:
            logger.error(f"No edge found with id {edge_id}.")

    @staticmethod
    def get_evolution_system_metrics():
        return {
            "fog_id": NodeState.get_current_node().id,
            "system_metrics": FogService.genetic_engine.evolution_system_metrics,
            "scope": MessageScope.EVOLUTION_SYSTEM_METRICS.value
        }
