# fog_node/genetic/genetic_engine.py

import random
import math
import os
import base64

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from deap import base, tools
from scipy.constants import Boltzmann
from shared.fed_node.node_state import NodeState
from shared.logging_config import logger
from fog_node.fog_resources_paths import FogResourcesPaths


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
        epochs = random.randint(1, 10)
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
        self.number_of_generations = number_of_generations
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
        self.fog_rabbitmq_host = None
        self.fog_edge_send_exchange = None
        self.edge_to_fog_queue = None

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

    def setup(self, fog_rabbitmq_host, fog_edge_send_exchange, edge_to_fog_queue):
        """
        Sets up the DEAP environment, including individuals, population and genetic operators.
        """

        self.fog_rabbitmq_host = fog_rabbitmq_host
        self.fog_edge_send_exchange = fog_edge_send_exchange
        self.edge_to_fog_queue = edge_to_fog_queue

        # toolbox for operators
        toolbox = base.Toolbox()

        # explicitly define the individual creation function
        def create_individual():
            return Individual.random_individual()

        # register individual and population creation
        toolbox.register("individual", create_individual)

        # register population creation
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # register evaluate function creation
        toolbox.register("evaluate", self.fitness_function)

        # genetic operators
        toolbox.register("mate", tools.cxOnePoint)  # crossover
        toolbox.register("mutate", tools.mutUniformInt, low=[1, 16, 1, 1, 1], up=[100, 120, 10, 20, 10], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)  # selection

        self.toolbox = toolbox

    def fitness_function(self, individual):
        """
        Synchronously computes the fitness value for an individual by sending a single HTTP POST
        request to each evaluation node's /execute-model-evaluation endpoint and waiting as long as needed.
        Missing responses are penalized.
        """

        # TODO: currently is using http request to request the metrics on the edge, in the future change to queue/ws

        logger.info("Starting fitness_function for individual: %s", individual)
        # Compute hyperparameters from the individual.
        learning_rate = individual[0] / 10000.0
        batch_size = individual[1]
        number_epochs = individual[2]
        patience = individual[3]
        fine_tune_layers = individual[4]
        logger.info("Hyperparameters: learning_rate=%s, batch_size=%s, epochs=%s, patience=%s, fine_tune_layers=%s",
                    learning_rate, batch_size, number_epochs, patience, fine_tune_layers)

        try:
            # Read the fog model file once.
            fog_model_file_path = os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH,
                                               FogResourcesPaths.FOG_MODEL_FILE_NAME)
            logger.info("Reading fog model file from: %s", fog_model_file_path)
            with open(fog_model_file_path, "rb") as model_file:
                model_bytes = model_file.read()
            model_file_base64 = base64.b64encode(model_bytes).decode("utf-8")
            logger.info("Successfully read fog model file. Encoded length: %d", len(model_file_base64))
        except Exception as e:
            logger.error("Error reading fog model file: %s", e)
            return float("inf")

        # Build a common payload template.
        payload_template = {
            "genetic_evaluation": True,
            "start_date": self.operating_data_date[0] if len(self.operating_data_date) == 2 else None,
            "current_date": self.operating_data_date[1] if len(self.operating_data_date) == 2 else
            self.operating_data_date[0],
            "is_cache_active": False,
            "model_type": None,
            "model_file": model_file_base64,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": number_epochs,
            "patience": patience,
            "fine_tune_layers": fine_tune_layers
        }
        logger.info("Payload template built with keys: %s", list(payload_template.keys()))

        # Get all evaluation nodes.
        evaluation_nodes = [child for child in NodeState.get_current_node().child_nodes if child.is_evaluation_node]
        logger.info("Found %d evaluation nodes.", len(evaluation_nodes))

        def send_request(node):
            payload = payload_template.copy()
            payload["child_id"] = node.id
            url = f"http://{node.ip_address}:{node.port}/edge/execute-model-evaluation"
            logger.info("Sending HTTP POST request to %s with child_id: %s", url, node.id)
            try:
                # No timeout is specified so that we wait as long as needed.
                r = requests.post(url, json=payload, timeout=None)
                logger.info("Received HTTP response from %s: status code %s", url, r.status_code)
                if 200 <= r.status_code < 300:
                    logger.info("Response from %s accepted.", url)
                    return r.json()
                else:
                    logger.error("HTTP error from %s: status code %s", url, r.status_code)
            except Exception as e:
                logger.error("HTTP request error to %s: %s", url, e)
            return None

        valid_responses = []
        with ThreadPoolExecutor(max_workers=len(evaluation_nodes)) as executor:
            futures = {executor.submit(send_request, node): node for node in evaluation_nodes}
            logger.info("Submitted requests to evaluation nodes; waiting for responses...")
            # Wait for all futures to complete (with no timeout)
            for future in as_completed(futures):
                node = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        logger.info("Received valid response from node %s", node.id)
                        valid_responses.append(res)
                    else:
                        logger.error("No valid response from node %s", node.id)
                except Exception as e:
                    logger.error("Exception while processing response from node %s: %s", node.id, e)

        logger.info("Collected %d valid responses out of required %d.", len(valid_responses),
                    self.number_of_evaluation_nodes)
        required = self.number_of_evaluation_nodes
        # If not enough responses are received, apply penalty defaults.
        if len(valid_responses) < required:
            logger.error("Not enough valid responses received; applying penalty for missing responses.")
            penalty = {"loss": 1e6, "mae": 1e6, "mse": 1e6, "rmse": 1e6, "r2": 1e6}
            penalty_response = {"metrics": {"before_training": penalty, "after_training": penalty}}
            for _ in range(required - len(valid_responses)):
                valid_responses.append(penalty_response)
            logger.info("Total responses after penalty: %d", len(valid_responses))

        weights = {"loss": 0.4, "mae": 0.3, "mse": 0.1, "rmse": 0.1, "r2": -0.1}

        def compute_weighted_score(metrics, weights):
            score = sum(weights[metric] * metrics[metric] for metric in weights)
            return score

        scores = []
        for resp in valid_responses:
            try:
                score_before = compute_weighted_score(resp["metrics"]["before_training"], weights)
                score_after = compute_weighted_score(resp["metrics"]["after_training"], weights)
                score = min(score_before, score_after)
                scores.append(score)
                logger.info("Computed score for response: before=%s, after=%s, chosen=%s", score_before, score_after,
                            score)
            except Exception as e:
                logger.error("Error computing score for a response: %s", e)
                scores.append(1e6)  # Penalty score
        final_score = sum(scores) / len(scores)
        logger.info("Final fitness score computed: %s", final_score)
        return final_score

    def evolve(self):
        logger.info("Starting evolution with population size: %d, generations: %d",
                    self.population_size, self.number_of_generations)
        if not self.toolbox:
            raise ValueError("Toolbox is not set up. Call setup() before evolve().")
        if self.current_population is None:
            self.current_population = self.toolbox.population(n=self.population_size)
            logger.info("Initialized population with %d individuals", len(self.current_population))

        best_individual = None
        for generation in range(self.number_of_generations):
            logger.info("Generation %d: Starting evaluation of population.", generation)
            invalid_ind = [ind for ind in self.current_population if not ind.fitness.valid or not ind.fitness.values]
            for ind in invalid_ind:
                try:
                    fit = self.toolbox.evaluate(ind)
                except Exception as eval_e:
                    logger.error("Error evaluating individual %s: %s", ind, eval_e)
                    fit = 1e6  # Penalty value
                ind.fitness.values = (fit,)

            try:
                best_individual = tools.selBest(self.current_population, 1)[0]
                best_fitness = best_individual.fitness.values[0]
            except Exception as sel_e:
                logger.error("Generation %d: Error selecting best individual: %s", generation, sel_e)
                break

            logger.info("Generation %d: Best fitness after evaluation: %s", generation, best_fitness)

            if best_fitness < self.best_fitness:
                self.best_fitness = best_fitness
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            logger.info("Generation %d: Stagnation counter = %d", generation, self.stagnation_counter)
            if self.stagnation_counter >= self.stagnation_limit:
                logger.info("Generation %d: Stagnation limit reached. Stopping evolution.", generation)
                break

            # Apply genetic operators
            offspring = []
            cloud_temperature = self.get_cloud_temperature()
            logger.info("Generation %d: Starting genetic operations with cloud_temperature = %s", generation,
                        cloud_temperature)
            for individual in self.current_population:
                try:
                    if self.should_skip_update(individual.fitness.values[0], cloud_temperature):
                        offspring.append(self.toolbox.clone(individual))
                        logger.info("Generation %d: Individual skipped update (cloned).", generation)
                    else:
                        mutant = self.toolbox.clone(individual)
                        if random.random() < 0.7:  # Crossover probability
                            partner = random.choice(self.current_population)
                            self.toolbox.mate(mutant, self.toolbox.clone(partner))
                            logger.info("Generation %d: Crossover performed.", generation)
                        if random.random() < 0.2:  # Mutation probability
                            self.toolbox.mutate(mutant)
                            logger.info("Generation %d: Mutation performed.", generation)
                        try:
                            del mutant.fitness.values  # Invalidate fitness
                        except Exception as e:
                            logger.error("Generation %d: Error invalidating fitness: %s", generation, e)
                        offspring.append(mutant)
                except Exception as op_e:
                    logger.error("Generation %d: Error processing individual genetic operator: %s", generation, op_e)
            logger.info("Generation %d: Genetic operations complete. Offspring count = %d", generation, len(offspring))
            self.current_population[:] = offspring
            logger.info("Generation %d: End of generation.", generation)

        logger.info("Evolution complete! Best individual: %s with fitness %s", best_individual, self.best_fitness)

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

        for ind in self.current_population:
            if not ind.fitness.values:
                ind.fitness.values = (1e6,)
        sorted_population = sorted(self.current_population, key=lambda ind: ind.fitness.values[0])
        return sorted_population[:k]

    def get_number_of_training_nodes(self):
        return self.number_of_training_nodes
