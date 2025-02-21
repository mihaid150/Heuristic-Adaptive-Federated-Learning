# fog_node/fog_service.py

import base64
import json
import os
import random
import threading
import time
from enum import Enum

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
        FogService.genetic_engine.set_number_of_evaluation_training_nodes(
            1,
            len(NodeState.get_current_node().child_nodes) - 1
        )
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
                        logger.info(f"Processing message for child id {child_id}.")

                        scheduler = FogService.get_fog_cooling_scheduler()
                        if scheduler is not None:
                            scheduler.reset()
                        else:
                            logger.error("Fog cooling scheduler is not initialized.")

                        FogService.orchestrate_training(message)
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
            evaluation_nodes_index = []
            while len(evaluation_nodes_index) < FogService.genetic_engine.number_of_evaluation_nodes:
                evaluation_nodes_index.append(random.randint(0, len(NodeState.get_current_node().child_nodes) - 1))

            for index, edge_node in enumerate(NodeState.get_current_node().child_nodes):
                if index in evaluation_nodes_index:
                    edge_node.is_evaluation_node = True

            if start_date is not None:
                FogService.genetic_engine.set_operating_data_date([start_date, current_date])
                # to not inherit population from a previous simulation
                delete_all_files_in_folder(SharedResourcesPaths.CACHE_FOLDER_PATH, "genetic")
            else:
                FogService.genetic_engine.set_operating_data_date([current_date])
            FogService.fog_service_state = FogServiceState.GENETIC_EVALUATION
            FogService.genetic_engine.evolve()

            # get top individuals for each fog child
            trainable_edges = [node for node in NodeState.get_current_node().child_nodes if not node.is_evaluation_node]
            top_individuals = FogService.genetic_engine.get_top_k_individuals(len(trainable_edges))

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

        trainable_edges = [node for node in current_node.child_nodes if not node.is_evaluation_node]
        for index, edge_node in enumerate(trainable_edges):
            individual = top_individuals[index]
            message = {
                "child_id": edge_node.id,
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
                    FogService.fog_cooling_scheduler.increment_stopping_score()
                    FogService.get_edge_model(body)
                    FogService.fog_cooling_scheduler.has_reached_stopping_condition_for_cooler()

                    if not FogService.fog_cooling_scheduler.is_fog_cooling_operational():
                        logger.info("Stopping queue listener as cooling process is complete.")
                        channel.stop_consuming()

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
    def get_edge_model(body):
        try:
            if not FogService.fog_cooling_scheduler.is_fog_cooling_operational():
                return
            message = json.loads(body.decode('utf-8'))
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
    def send_fog_model_to_cloud():
        logger.info("Running the sending of the fog model to cloud.")

        fog_model_file_path = os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH,
                                           FogResourcesPaths.FOG_MODEL_FILE_NAME)
        with open(fog_model_file_path, "rb") as model_file:
            model_bytes = model_file.read()
            model_file_base64 = base64.b64encode(model_bytes).decode("utf-8")

        message = {
            "fog_id": NodeState.get_current_node().id,
            "lambda_prev": read_lambda_prev(),
            "model_file": model_file_base64
        }

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
                logger.info("Fog has sent fog model to cloud node.")
            except Exception as e:
                logger.error(f"Error sending fog model to cloud: {e}. Retrying in 5 seconds...")
                time.sleep(5)

        # Restart the evaluation/training node assignment
        for node in NodeState.get_current_node().child_nodes:
            node.is_evaluation_node = False

        FogService.fog_service_state = FogServiceState.IDLE
