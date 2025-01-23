import random
import math
import os
import base64
import pika
import json
from deap import base, tools
from scipy.constants import Boltzmann
from fed_node.node_state import NodeState
from fog_node.fog_resources_paths import FogResourcesPaths
from fog_node.fog_service import FogService


class FitnessMin(base.Fitness):
    """
    Represents a fitness object with minimization objective.
    """
    weights = (-1.0,)


class Individual(list):
    """
    Represents an individual in the genetic algorithm with predefined attributes.
    """
    def __init__(self, learning_rate, batch_size, epochs, patience, fine_tune_layers):
        super().__init__([learning_rate, batch_size, epochs, patience, fine_tune_layers])
        self.fitness = FitnessMin()

    @staticmethod
    def random_individual():
        """
        Creates a random individual with attributes initialized within their respective ranges.
        """
        learning_rate = random.randint(1, 100)
        batch_size = random.randint(16, 128)
        epochs = random.randint(1, 100)
        patience = random.randint(1, 20)
        fine_tune_layers = random.randint(1, 10)
        return Individual(learning_rate, batch_size, epochs, patience, fine_tune_layers)


class GeneticEngine:
    def __init__(self, population_size, number_of_generations, stagnation_limit):
        """
        Initializes the Genetic Engine
        :param population_size: Size of the population.
        :param number_of_generations: Number of generations to run.
        :param stagnation_limit: Limit of stagnation for early stopping
        """
        self.population_size = population_size
        self.number_of_generation = number_of_generations
        self.stagnation_limit = stagnation_limit
        self.toolbox = None
        self.best_fitness = float("inf")
        self.stagnation_counter = 0
        self.boltzmann_constant = Boltzmann
        self.additional_factor = 1e23
        self.current_population = None
        self.operating_data_date = []
        self.number_of_evaluation_nodes = None
        self.number_of_training_nodes = None
        self.genetic_evaluation_strategy = None

    @staticmethod
    def get_cloud_temperature():
        return random.uniform(1, 100)

    def set_operating_data_date(self, dates):
        self.operating_data_date = dates

    def clear_operating_data_date(self):
        self.operating_data_date = []

    def set_number_of_evaluation_training_nodes(self, no_eval_nodes, no_training_nodes):
        self.number_of_evaluation_nodes = no_eval_nodes
        self.number_of_training_nodes = no_training_nodes

    def should_skip_update(self, fitness, cloud_temperature):
        """
        Determines whether an individual should skip genetic operations based on Boltzmann probability.
        :param fitness: The fitness value of the individual.
        :param cloud_temperature: The temperature value retrieved from the cloud.
        :return: True if the individual should skip the update; False otherwise.
        """
        probability = math.exp(-fitness / (self.boltzmann_constant * self.additional_factor * cloud_temperature))
        return random.random() > probability

    def setup(self):
        """
        Sets up the DEAP environment, including individuals, population and genetic operators.
        """

        # toolbox for operators
        toolbox = base.Toolbox()

        # explicitly define the individual creation function
        def create_individual():
            return Individual.random_individual()

        # register individual and population creation
        toolbox.register("individual", create_individual)

        # register population creation
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # register evaluate function creation
        toolbox.register("evaluate", self.fitness_function)

        # genetic operators
        toolbox.register("mate", tools.cxOnePoint)  # crossover
        toolbox.register("mutate", tools.mutUniformInt, low=[1, 16, 1, 1, 1], up=[100, 120, 100, 20, 10], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)  # selection

        self.toolbox = toolbox

    def fitness_function(self, individual):
        """
        Computes the fitness of an individual.
        :param individual: A list of hyperparameters [learning_rate, batch_size, epochs, patience, fine_tune_layers].
        :return: The fitness value, lower being better.
        """
        learning_rate = individual[0] / 10000.0
        batch_size = individual[1]
        number_epochs = individual[2]
        patience = individual[3]
        fine_tune_layers = individual[4]

        connection = pika.BlockingConnection(pika.ConnectionParameters(host=FogService.FOG_RABBITMQ_HOST))
        channel = connection.channel()

        # create a temporary queue for responses
        result_queue = channel.queue_declare(queue='', exclusive=True)
        response_queue = result_queue.method.queue

        # generate a unique correlation ID for this fitness evaluation
        correlation_id = str(random.randint(1, 100000))

        for node in [child for child in NodeState.get_current_node().child_nodes if child.is_evaluation_node]:
            fog_model_file_path = os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH,
                                               FogResourcesPaths.FOG_MODEL_FILE_NAME)
            with open(fog_model_file_path, "rb") as model_file:
                model_bytes = model_file.read()
                model_file_base64 = base64.b64encode(model_bytes).decode("utf-8")

            message = {
                "child_id": node.id,
                "genetic_evaluation": True,
                "dates": self.operating_data_date,
                "is_cache_active": False,
                "model_type": None,  # to be determined
                "model_file": model_file_base64,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": number_epochs,
                "patience": patience,
                "fine_tune_layers": fine_tune_layers
            }

            channel.basic_publish(
                exchange="",
                routing_key=FogService.FOG_EDGE_SEND_QUEUE,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    reply_to=response_queue,
                    correlation_id=correlation_id,
                )
            )

        # collect responses
        responses = []

        def on_response(ch, method, properties, body):
            # check if the response matches the correlation id
            if properties.correlation_id == correlation_id:
                responses.append(json.loads(body.decode("utf-8")))

        # start consuming the response queue
        channel.basic_consume(queue=response_queue, on_message_callback=on_response, auto_ack=True)

        print("Waiting for responses from edge nodes...")
        while len(responses) < self.number_of_evaluation_nodes:
            connection.process_data_events()  # process incoming messages

        connection.close()

        if not responses:
            print("No responses received within the timeout")
            return float("inf")

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

        scores = [
            min(
                compute_weighted_score(response["metrics"]["before_training"], weights),
                compute_weighted_score(response["metrics"]["after_training"], weights)
            )
            for response in responses
        ]

        return sum(scores) / len(responses)

    def evolve(self):
        """
        Executes the genetic algorithm
        """
        if not self.toolbox:
            raise ValueError("Toolbox is not set up. Call setup() before evolve().")

        # initialize population if it's the first call to evolve
        if self.current_population is None:
            self.current_population = self.toolbox.population(n=self.population_size)

        best_individual = None
        for generation in range(self.number_of_generation):
            # evaluate the entire population
            fitnesses = list(map(self.toolbox.evaluate, self.current_population))

            for individual, fitness in zip(self.current_population, fitnesses):
                individual.fitness.values = fitness

            # find the best individual
            best_individual = tools.selBest(self.current_population, 1)[0]
            best_fitness = best_individual.fitness.values[0]

            if best_fitness < self.best_fitness:
                self.best_fitness = best_fitness
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            print(f"Generation {generation}: Best fitness = {best_fitness}")

            if self.stagnation_counter >= self.stagnation_limit:
                print("Stagnation limit reached. Stopping early.")
                break

            # Apply genetic operators based on Boltzmann probability
            offspring = []
            cloud_temperature = self.get_cloud_temperature()
            for individual in self.current_population:
                if self.should_skip_update(individual.fitness.values[0], cloud_temperature):
                    offspring.append(self.toolbox.clone(individual))
                else:
                    mutant = self.toolbox.clone(individual)
                    if random.random() < 0.7:  # Crossover probability
                        partner = random.choice(self.current_population)
                        self.toolbox.mate(mutant, self.toolbox.clone(partner))
                    if random.random() < 0.2:  # Mutation probability
                        self.toolbox.mutate(mutant)
                    del mutant.fitness.values  # Invalidate fitness
                    offspring.append(mutant)

            # Replace population with offspring
            self.current_population[:] = offspring

        print(f"Evolution complete! Best individual: {best_individual} with fitness {self.best_fitness}")

    def get_top_k_individuals(self, k):
        """
        Extracts the top k performing individuals from the current population.
        :param k: Number of top-performing individuals to extract.
        :return:  A list of top k individuals sorted by fitness (ascending order).
        """
        if self.current_population is None:
            return ValueError("Population is empty. Ensure evolution has been run before calling this method.")

        if k > len(self.current_population):
            raise ValueError(f"Requested top {k} individuals, but the population size is {len(self.current_population)}"
                             f".")

        # sort the population based on fitness in ascending order
        sorted_population = sorted(self.current_population, key=lambda individual: individual.fitness.values[0])

        return sorted_population[:k]

    def get_number_of_training_nodes(self):
        return self.number_of_training_nodes
