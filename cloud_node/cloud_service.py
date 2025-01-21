import base64
import json
import os
import time

import pika
from fed_node.node_state import NodeState
from model_manager import create_initial_lstm_model
from cloud_resources_paths import CloudResourcesPaths
from fastapi import HTTPException
from cloud_cooling_scheduler import CloudCoolingScheduler


class CloudService:
    RABBITMQ_HOST = "localhost"
    SEND_QUEUE = "cloud_to_fog_queue"
    RECEIVE_QUEUE = "fog_to_cloud_queue"

    cooling_scheduler = CloudCoolingScheduler()

    @staticmethod
    def init_rabbitmq():
        """
        Initialize RabbitMQ connection and declare queues.
        """
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=CloudService.RABBITMQ_HOST))
        channel = connection.channel()

        # declare the queues for sending and receiving messages
        channel.queue_declare(queue=CloudService.SEND_QUEUE)
        channel.queue_declare(queue=CloudService.RECEIVE_QUEUE)
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
    def init_process(is_cache_active: bool, genetic_evaluation_strategy: str, model_type: str):
        # create the LSTM model
        cloud_model = create_initial_lstm_model()

        # save the model locally
        cloud_model_file_path = os.path.join(CloudResourcesPaths.CLOUD_MODEL_FOLDER_PATH,
                                             CloudResourcesPaths.CLOUD_MODEL_FILE_NAME)
        cloud_model.save(cloud_model_file_path)

        # read the saved file in bytes format
        with open(cloud_model_file_path, "rb") as model_file:
            model_bytes = model_file.read()
            encoded_model = base64.b64encode(model_bytes).decode('utf-8')

        # prepare the payload
        payload = {
            "is_cache_active": is_cache_active,
            "genetic_evaluation_strategy": genetic_evaluation_strategy,
            "model_type": model_type,
            "model_file": encoded_model
        }

        # get the current cloud node
        cloud_node = NodeState.get_current_node()
        if not cloud_node:
            raise ValueError("Current not is not initialized. Go back and initialized it first.")

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=CloudService.RABBITMQ_HOST))
        channel = connection.channel()

        # send the payload and the model to each child
        for child_node in cloud_node.child_nodes:
            message = {
                "child_id": child_node.id,
                "url": f"http://{child_node.ip_address}:{child_node.port}/fog/get-cloud-model",
                **payload,
            }

            channel.basic_publish(exchange="", routing_key=CloudService.SEND_QUEUE, body=str(message))
            print(f"Request sent to queue for child {child_node.name} ({child_node.ip_address}:{child_node.port})")

        # initialize the cloud cooling scheduler
        CloudService.cooling_scheduler.start_cooling()

        connection.close()

    @staticmethod
    def get_fog_model(timeout: int = 30):
        """
        Checks the receiving queue for messages from all children.
        Processes available requests if not all are received within the threshold time.
        :return:
        """
        try:
            cloud_node = NodeState.get_current_node()
            if not cloud_node:
                raise ValueError("Current node is not initialized.")

            # get the list of expected child ids
            expected_children = {child.id for child in cloud_node.child_nodes}

            # connect to rabbitmq
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=CloudService.RABBITMQ_HOST))
            channel = connection.channel()

            print("Listening for messages from fog nodes...")
            received_messages = {}
            start_time = time.time()

            while CloudService.cooling_scheduler.is_cooling_operational and time.time() - start_time < timeout:
                method_frame, header_frame, body = channel.basic_get(queue=CloudService.RECEIVE_QUEUE, auto_ack=True)
                if body:
                    message = json.loads(body.decode('utf-8'))
                    child_id = message.get("child_id")
                    if child_id in expected_children:
                        received_messages[child_id] = message
                        print(f"Received message from fog {child_id}: {message}")
                if len(expected_children) == len(expected_children):
                    print("Received responses from all children.")
                    break
            connection.close()

            # handle the received messages
            if received_messages:
                for child_id, message in received_messages.items():
                    print(f"Processing message from child {child_id}: {message}")
            else:
                print(f"No messages received from any child nodes within the timeout. We keep the current cloud model.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
