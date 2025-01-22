import pika
import json
import base64
import os
from fed_node.node_state import NodeState
from edge_node.edge_resources_paths import EdgeResourcesPaths
from model_manager import create_initial_lstm_model


class EdgeService:
    FOG_RABBITMQ_HOST = "fog-rabbitmq-host"
    FOG_EDGE_RECEIVE_QUEUE = "fog_to_edge_queue"
    EDGE_FOG_SEND_QUEUE = "edge_to_fog_queue"

    @staticmethod
    def init_rabbitmq() -> None:
        """
        Initialize RabbitMQ connection and declare queues.
        """
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=EdgeService.FOG_RABBITMQ_HOST))
        channel = connection.channel()

        # declare the receiving queue for listening and sending messages from/to the fog
        channel.queue_declare(queue=EdgeService.FOG_EDGE_RECEIVE_QUEUE)
        channel.queue_declare(queue=EdgeService.EDGE_FOG_SEND_QUEUE)
        connection.close()

    @staticmethod
    def init_process() -> None:
        EdgeService.init_rabbitmq()

    @staticmethod
    def listen_to_fog_receiving_queue() -> None:
        """
        Start listening to the RabbitMQ queue for messages from the fog node.
        """
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=EdgeService.FOG_EDGE_RECEIVE_QUEUE))
        channel = connection.channel()

        def callback(body):
            """
            RabbitMQ callback to process messages received from the fog queue.
            :param body: The actual message body.
            """

            try:
                # deserialize the message
                message = json.loads(body.decode("utf-8"))
                child_id = message.get("child_id")

                # check of the message is intended for this edge node
                if child_id == NodeState.get_current_node().id:
                    EdgeService.process_received_message(message)
                else:
                    print(f"Ignoring message for child id {child_id}.")
            except Exception as e:
                print(f"Error processing message: {str(e)}")

        channel.basic_consume(queue=EdgeService.FOG_EDGE_RECEIVE_QUEUE, on_message_callback=callback, auto_ack=True)
        print("Listening for messages from the fog...")
        channel.start_consuming()

    @staticmethod
    def process_received_message(message):
        """
        Process a received message from cloud node.
        :param message: The received message containing the payload and model file.
        """
        try:
            start_date = message.get("start_date")
            end_date = message.get("end_date")
            is_cache_active = message.get("is_cache_active")
            genetic_evaluation_strategy = message.get("genetic_evaluation_strategy")
            model_type = message.get("model_type")
            model_file_base64 = message.get("model_file")

            if not model_file_base64:
                raise ValueError("Model file is missing in the received message.")

            # decode the base64 model file
            model_file_bytes = base64.b64decode(model_file_base64)

            # save the model file locally
            edge_model_file_path = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH,
                                                EdgeResourcesPaths.EDGE_MODEL_FILE_NAME)

            os.makedirs(EdgeResourcesPaths.MODELS_FOLDER_PATH, exist_ok=True)  # ensure the directory exists
            with open(edge_model_file_path, "wb") as model_file:
                model_file.write(model_file_bytes)

            print(f"Received fog model saved at: {edge_model_file_path}")

            # handle the received payload

        except Exception as e:
            print(f"Error processing the received message: {str(e)}")