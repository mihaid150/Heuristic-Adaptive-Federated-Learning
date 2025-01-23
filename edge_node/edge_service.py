import pika
import json
import base64
import os
from fed_node.node_state import NodeState
from edge_node.edge_resources_paths import EdgeResourcesPaths
from model_manager import pretrain_edge_model, retrain_edge_model


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

        def callback(ch, method, properties, body):
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
                    EdgeService.execute_model_training_evaluation(message)
                    ch.basic_ack(delivery_tag=method.delivery_tag)  # ack rabbitmq to remove message only if processed
                else:
                    print(f"Ignoring message for child id {child_id}.")
                    # no ack needed here, let other edges to handle it
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                # in case of error we must negatively acknowledge the message and requeue it
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        channel.basic_consume(queue=EdgeService.FOG_EDGE_RECEIVE_QUEUE, on_message_callback=callback, auto_ack=False)
        print("Listening for messages from the fog...")
        channel.start_consuming()

    @staticmethod
    def execute_model_training_evaluation(message):
        """
        Process a received message from cloud node.
        :param message: The received message containing the payload and model file.
        """
        try:
            genetic_evaluation = message.get("genetic_evaluation")
            dates = message.get("start_date")
            is_cache_active = message.get("is_cache_active")
            model_type = message.get("model_type")
            model_file_base64 = message.get("model_file")
            learning_rate = message.get("learning_rate")
            batch_size = message.get("batch_size")
            epochs = message.get("epochs")
            patience = message.get("patience")
            fine_tune_layers = message.get("fine_tune_layers")

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

            if len(dates) == 2:
                # pretraining
                metrics = pretrain_edge_model(edge_model_file_path, dates[0], dates[1], learning_rate, batch_size,
                                              epochs, patience, fine_tune_layers)

                pretrained_model_file_path = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH,
                                                          EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME)

                with open(pretrained_model_file_path, "rb") as model_file:
                    model_bytes = model_file.read()
                    model_file_base64 = base64.b64encode(model_bytes).decode("utf-8")

                message = {
                    "child_id": NodeState.get_current_node().parent_node.id,
                    "metrics": metrics
                }

                if not genetic_evaluation:
                    message["model_file"] = model_file_base64
                    message["edge_id"] = NodeState.get_current_node().id,
                else:
                    message["genetic_evaluation"] = True

                connection = pika.BlockingConnection(pika.ConnectionParameters(host=EdgeService.FOG_RABBITMQ_HOST))
                channel = connection.channel()
                channel.basic_publish(exchange="", routing_key=EdgeService.EDGE_FOG_SEND_QUEUE, body=json.dumps(message))
                connection.close()

                if not genetic_evaluation:
                    print(f"Pretraining complete. Sent updated model and metrics to parent node with ID "
                          f"{NodeState.get_current_node().parent_node.id}.")
                else:
                    print(f"Pretraining genetic evaluation complete. Sent metrics to parent node with ID "
                          f"{NodeState.get_current_node().parent_node.id}")
            else:
                # retraining
                metrics = retrain_edge_model(edge_model_file_path, dates[0], learning_rate, batch_size, epochs,
                                             patience, fine_tune_layers)

                weights = {
                    "loss": 0.4,  # higher weight for loss
                    "mae": 0.3,  # medium weight for mae
                    "mse": 0.1,  # lower weight for mse since it's redundant loss
                    "rmse": 0.1,  # lower weight for rmse
                    "r2": -0.1,  # negative weight because higher r2 is better
                }

                def compute_weighted_score(metrics_for_score, weights_for_score):
                    score = 0
                    for metric, weight in weights_for_score.items():
                        score += weight * metrics_for_score[metric]
                    return score

                score_before = compute_weighted_score(metrics["before_training"], weights)
                score_after = compute_weighted_score(metrics["after_training"], weights)

                # decide which model to send based on the weighted scores, choose the one with less score
                if not genetic_evaluation and score_after < score_before:
                    retrained_model_file_path = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH,
                                                             EdgeResourcesPaths.RETRAINED_EDGE_MODEL_FILE_NAME)

                    with open(retrained_model_file_path, "rb") as model_file:
                        model_bytes = model_file.read()
                        model_file_base64 = base64.b64encode(model_bytes).decode("utf-8")

                    message = {
                        "child_id": NodeState.get_current_node().parent_node.id,
                        "edge_id": NodeState.get_current_node().id,
                        "metrics": metrics,
                        "model_file": model_file_base64
                    }

                    print(f"Retrained model selected with score {score_after:.4f} (better than {score_before:.4f}).")
                elif not genetic_evaluation and score_after > score_before:
                    with open(edge_model_file_path, "rb") as model_file:
                        model_bytes = model_file.read()
                        model_file_base64 = base64.b64encode(model_bytes).decode("utf-8")

                    message = {
                        "child_id": NodeState.get_current_node().parent_node.id,
                        "edge_id": NodeState.get_current_node().id,
                        "metrics": metrics,
                        "model_file": model_file_base64
                    }

                    print(f"Original model selected with score {score_before:.4f} (better than {score_after:.4f}).")
                else:
                    message = {
                        "child_id": NodeState.get_current_node().parent_node.id,
                        "genetic_evaluation": True,
                        "metrics": metrics,
                    }
                    print(f"Retraining genetic evaluation with before score {score_before:.4f} and after "
                          f"score {score_after:.4f}).")

                connection = pika.BlockingConnection(pika.ConnectionParameters(host=EdgeService.FOG_RABBITMQ_HOST))
                channel = connection.channel()
                channel.basic_publish(exchange="", routing_key=EdgeService.EDGE_FOG_SEND_QUEUE,
                                      body=json.dumps(message))
                connection.close()
                if not genetic_evaluation:
                    print(f"Retraining complete. Sent model and metrics to parent node with ID "
                          f"{NodeState.get_current_node().parent_node.id}.")
                else:
                    print(f"Retraining genetic evaluation complete. Sent metrics to parent node with ID "
                          f"{NodeState.get_current_node().parent_node.id}")

        except Exception as e:
            print(f"Error processing the received message: {str(e)}")