import base64
import json
import os

import threading
import pika
from fed_node.node_state import NodeState
from model_manager import create_initial_lstm_model, aggregate_fog_models
from cloud_resources_paths import CloudResourcesPaths
from cloud_cooling_scheduler import CloudCoolingScheduler


class CloudService:

    CLOUD_RABBITMQ_HOST = "cloud-rabbitmq-host"
    CLOUD_FOG_SEND_QUEUE = "cloud_to_fog_queue"
    FOG_CLOUD_RECEIVE_QUEUE = "fog_to_cloud_queue"

    cloud_cooling_scheduler = CloudCoolingScheduler()
    received_fog_messages = {}

    @staticmethod
    def init_rabbitmq():
        """
        Initialize RabbitMQ connection and declare queues.
        """
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=CloudService.CLOUD_RABBITMQ_HOST))
        channel = connection.channel()

        # declare the queues for sending and receiving messages
        channel.queue_declare(queue=CloudService.CLOUD_FOG_SEND_QUEUE)
        channel.queue_declare(queue=CloudService.FOG_CLOUD_RECEIVE_QUEUE)
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
    def init_process(start_date: str, end_date: str, is_cache_active: bool, genetic_evaluation_strategy: str,
                     model_type: str):
        CloudService.init_rabbitmq()

        # create the LSTM model
        cloud_model = create_initial_lstm_model()

        # save the model locally
        cloud_model_file_path = os.path.join(CloudResourcesPaths.MODELS_FOLDER_PATH,
                                             CloudResourcesPaths.CLOUD_MODEL_FILE_NAME)
        cloud_model.save(cloud_model_file_path)

        # read the saved file in bytes format
        with open(cloud_model_file_path, "rb") as model_file:
            model_bytes = model_file.read()
            encoded_model = base64.b64encode(model_bytes).decode('utf-8')

        # prepare the payload
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "is_cache_active": is_cache_active,
            "genetic_evaluation_strategy": genetic_evaluation_strategy,
            "model_type": model_type,
            "model_file": encoded_model
        }

        # get the current cloud node
        cloud_node = NodeState.get_current_node()
        if not cloud_node:
            raise ValueError("Current not is not initialized. Go back and initialized it first.")

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=CloudService.CLOUD_RABBITMQ_HOST))
        channel = connection.channel()

        # send the payload and the model to each child
        for child_node in cloud_node.child_nodes:
            message = {
                "child_id": child_node.id,
                **payload,
            }

            channel.basic_publish(exchange="", routing_key=CloudService.CLOUD_FOG_SEND_QUEUE, body=str(message))
            print(f"Request sent to queue for child {child_node.name} ({child_node.ip_address}:{child_node.port})")

        # initialize the cloud cooling scheduler
        CloudService.cloud_cooling_scheduler.start_cooling()

        listener_thread = threading.Thread(target=CloudService.listen_to_receive_fog_queue, daemon=True)
        listener_thread.start()

        connection.close()

    @staticmethod
    def get_fog_model(body):
        """
        Process messages from the queue whenever they are received.
        :param body: The actual message body.
        """
        try:
            if not CloudService.cloud_cooling_scheduler.is_cloud_cooling_operational():
                print("Cooling process has finished. Ignoring further messages.")
                aggregate_fog_models(CloudService.received_fog_messages)
                return

            # Deserialize the message
            message = json.loads(body.decode('utf-8'))
            child_id = message.get("fog_id")

            # Process the message
            if child_id not in CloudService.received_fog_messages:
                CloudService.process_received_messages(message, child_id)

        except Exception as e:
            print(f"Error processing message: {str(e)}")

    @staticmethod
    def process_received_messages(message: dict, child_id: int) -> None:
        fog_model_file_bytes = base64.b64decode(message["model_file"])
        lambda_prev_value = message.get("lambda_prev")

        fog_model_file_name = CloudResourcesPaths.FOG_MODEL_FILE_NAME.format(child_id=child_id)
        fog_model_file_path = os.path.join(CloudResourcesPaths.MODELS_FOLDER_PATH, fog_model_file_name)

        with open(fog_model_file_path, "wb") as fog_model_file:
            fog_model_file.write(fog_model_file_bytes)

        print(f"Received message from fog {child_id}. The fog model was saved successfully.")

        CloudService.received_fog_messages[child_id] = {"fog_model_file_path": fog_model_file_path,
                                                        "lambda_prev": lambda_prev_value}

        if len(CloudService.received_fog_messages) == len(NodeState.get_current_node().child_nodes):
            print("Received messages from all fogs. Stopping the cooling process.")
            CloudService.cloud_cooling_scheduler.stop_cooling()
            aggregate_fog_models(CloudService.received_fog_messages)

    @staticmethod
    def listen_to_receive_fog_queue():
        """
        Start listening to the RabbitMQ queue for incoming messages.
        This stops automatically when the cooling scheduler stops.
        """

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=CloudService.CLOUD_RABBITMQ_HOST))
        channel = connection.channel()

        def callback(ch, method, properties, body):
            """
            RabbitMQ callback to process messages
            """
            CloudService.get_fog_model(body)

            # check if cooling has stopped
            if not CloudService.cloud_cooling_scheduler.is_cooling_operational():
                print("Stopping queue listener as cooling process is complete.")
                channel.stop_consuming()

        print("Listening for messages from fog nodes...")
        channel.basic_consume(queue=CloudService.FOG_CLOUD_RECEIVE_QUEUE, on_message_callback=callback, auto_ack=True)
        channel.start_consuming()