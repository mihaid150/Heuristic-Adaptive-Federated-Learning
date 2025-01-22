import base64
import json
import os
import threading
import pika
from fed_node.node_state import NodeState
from fog_resources_paths import FogResourcesPaths
from genetics.genetic_engine import GeneticEngine


class FogService:
    CLOUD_RABBITMQ_HOST = "cloud-rabbitmq-host"
    CLOUD_FOG_RECEIVE_QUEUE = "cloud_to_fog_queue"
    FOG_CLOUD_SEND_QUEUE = "fog_to_cloud_queue"

    FOG_RABBITMQ_HOST = "fog-rabbitmq-host"
    FOG_EDGE_SEND_QUEUE = "fog_to_edge_queue"
    EDGE_FOG_RECEIVE_QUEUE = "edge_to_fog_queue"
    genetic_engine = GeneticEngine(5, 5, 2)

    @staticmethod
    def init_process() -> None:
        FogService.init_rabbitmq()
        FogService.genetic_engine.setup()
        listener_thread = threading.Thread(target=FogService.listen_to_cloud_receiving_queue, daemon=True)
        listener_thread.start()

    @staticmethod
    def init_rabbitmq() -> None:
        """
        Initialize RabbitMQ connection and declare queues.
        """
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=FogService.CLOUD_RABBITMQ_HOST))
        channel = connection.channel()

        # declare the receiving queue for listening and sending messages from/to the cloud
        channel.queue_declare(queue=FogService.CLOUD_FOG_RECEIVE_QUEUE)
        channel.queue_declare(queue=FogService.FOG_CLOUD_SEND_QUEUE)
        connection.close()

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=FogService.FOG_RABBITMQ_HOST))
        channel = connection.channel()

        # declare the receiving queue for listening and sending messages from/to the edge
        channel.queue_declare(queue=FogService.FOG_EDGE_SEND_QUEUE)
        channel.queue_declare(queue=FogService.EDGE_FOG_RECEIVE_QUEUE)
        connection.close()

    @staticmethod
    def listen_to_cloud_receiving_queue() -> None:
        """
        Start listening to the RabbitMQ queue for messages from the cloud node.
        """
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=FogService.CLOUD_RABBITMQ_HOST))
        channel = connection.channel()

        def callback(body):
            """
            RabbitMQ callback to process messages received from the cloud queue.
            :param body: The actual message body.
            """

            try:
                # deserialize the message
                message = json.loads(body.decode("utf-8"))
                child_id = message.get("child_id")

                # check if the message is intended for this fog node
                if child_id == NodeState.get_current_node().id:
                    FogService.orchestrate_training(message)
                else:
                    print(f"Ignoring message for child id {child_id}.")
            except Exception as e:
                print(f"Error processing message: {str(e)}")

        channel.basic_consume(queue=FogService.CLOUD_FOG_RECEIVE_QUEUE, on_message_callback=callback, auto_ack=True)
        print("Listening for messages from the cloud...")
        channel.start_consuming()

    @staticmethod
    def orchestrate_training(message):
        """
        Process a received message from cloud node.
        :param message: The received message containing the payload and model file.
        """
        try:
            start_date = message.get("start_date")
            current_date = message.get("current_date")
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

            print(f"Received cloud model saved at: {fog_model_file_path}")

            # handle the received payload
            print(f"Processing with start_date: {start_date}, current_date: {current_date}, cache_active: "
                  f"{is_cache_active}, strategy: {genetic_evaluation_strategy}")

            # run the genetic engine
            if start_date is not None:
                FogService.genetic_engine.set_evaluation_data_date([start_date, current_date])
            else:
                FogService.genetic_engine.set_evaluation_data_date([current_date])
            FogService.genetic_engine.evolve()

            # get top individuals for each fog child
            top_individuals = FogService.genetic_engine.get_top_k_individuals(
                len(NodeState.get_current_node().child_nodes))

            FogService.forward_model_to_edges(model_file_base64, start_date, current_date, is_cache_active,
                                              genetic_evaluation_strategy, model_type, top_individuals)

        except Exception as e:
            print(f"Error processing the received message: {str(e)}")

    @staticmethod
    def forward_model_to_edges(model_file_base64: str, start_date: str, current_date: str, is_cache_active: bool,
                               genetic_evaluation_strategy: str, model_type: str, top_individuals) -> None:
        """
        Forward the received model and payload to all child edge nodes.
        """
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=FogService.FOG_RABBITMQ_HOST))
        channel = connection.channel()

        current_node = NodeState.get_current_node()
        if not current_node:
            raise ValueError("Current fog node is not initialized.")

        for index, edge_node in enumerate(current_node.child_nodes):
            individual = top_individuals[index]
            message = {
                "child_id": edge_node.id,
                "start_date": start_date,
                "current_date": current_date,
                "is_cache_active": is_cache_active,
                "genetic_evaluation_strategy": genetic_evaluation_strategy,
                "model_type": model_type,
                "model_file": model_file_base64,
                "learning_rate": individual[0],
                "batch_size": individual[1],
                "epochs": individual[2],
                "patience": individual[3],
                "fine_tune_layers": individual[4]
            }

            channel.basic_publish(exchange="", routing_key=FogService.FOG_EDGE_SEND_QUEUE, body=json.dumps(message))
            print(f"Forwarded model to edge {edge_node.name} ({edge_node.ip_address}:{edge_node.port})")

        connection.close()
