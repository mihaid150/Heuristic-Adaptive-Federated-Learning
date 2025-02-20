# fog_node/genetic/genetic_engine.py

import random
import math
import os
import base64
import json
from typing import Any

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from deap import base, tools
from scipy.constants import Boltzmann
from shared.fed_node.node_state import NodeState
from shared.logging_config import logger
from fog_node.fog_resources_paths import FogResourcesPaths
from shared.shared_resources_paths import SharedResourcesPaths


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
        self.toolbox: Any = base.Toolbox()
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

        self.crossover_probability = 0.7
        self.mutation_probability = 0.2

        self.min_crossover_probability = 0.3
        self.max_crossover_probability = 0.9
        self.min_mutation_probability = 0.1
        self.max_mutation_probability = 0.6

        self.fog_model_file_path = os.path.join(FogResourcesPaths.MODELS_FOLDER_PATH,
                                                FogResourcesPaths.FOG_MODEL_FILE_NAME)
        self.genetic_population_file_path = os.path.join(SharedResourcesPaths.CACHE_FOLDER_PATH,
                                                         FogResourcesPaths.GENETIC_POPULATION_FILE_NAME)

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

        # explicitly define the individual creation function
        def create_individual():
            return Individual.random_individual()

        # register individual and population creation
        self.toolbox.register("individual", create_individual)

        # register population creation
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # register evaluate function creation
        self.toolbox.register("evaluate", self.fitness_function)

        # genetic operators
        self.toolbox.register("mate", tools.cxOnePoint)  # crossover
        self.toolbox.register("mutate", tools.mutUniformInt, low=[1, 16, 1, 1, 1], up=[100, 120, 10, 20, 10], indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # selection

    def fitness_function(self, individual):
        """
        Synchronously computes the fitness value for an individual by sending a single HTTP POST
        request to each evaluation node's /execute-model-evaluation endpoint and waiting as long as needed.
        Missing responses are penalized.
        """

        # TODO: currently is using http request to request the metrics on the edge, in the future change to queue/ws

        logger.info(f"Starting fitness_function for individual: {individual}")
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

            logger.info(f"Reading fog model file from: {self.fog_model_file_path}")
            with open(self.fog_model_file_path, "rb") as model_file:
                model_bytes = model_file.read()
            model_file_base64 = base64.b64encode(model_bytes).decode("utf-8")
            logger.info(f"Successfully read fog model file. Encoded length: {len(model_file_base64)}")
        except Exception as e:
            logger.error(f"Error reading fog model file: {e}")
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
        logger.info(f"Payload template built with keys: {list(payload_template.keys())}")

        # Get all evaluation nodes.
        evaluation_nodes = [child for child in NodeState.get_current_node().child_nodes if child.is_evaluation_node]
        logger.info(f"Found {len(evaluation_nodes)} evaluation nodes.")

        def send_request(node):
            payload = payload_template.copy()
            payload["child_id"] = node.id
            url = f"http://{node.ip_address}:{node.port}/edge/execute-model-evaluation"
            logger.info(f"Sending HTTP POST request to {url} with child_id: {node.id}.")
            try:
                # No timeout is specified so that we wait as long as needed.
                r = requests.post(url, json=payload, timeout=None)
                logger.info(f"Received HTTP response from {url}: status code {r.status_code}")
                if 200 <= r.status_code < 300:
                    logger.info(f"Response from {url} accepted.")
                    return r.json()
                else:
                    logger.error(f"HTTP error from {url}: status code {r.status_code}")
            except Exception as e1:
                logger.error(f"HTTP request error to {url}: {e1}")
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

        def compute_weighted_score(metrics, local_weights):
            return sum(local_weights[metric] * metrics[metric] for metric in local_weights)

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
            self.load_population_from_json()

        previous_best = self.best_fitness

        best_individual = None
        for generation in range(self.number_of_generations):
            logger.info("Generation %d: Starting evaluation of population.", generation)
            for ind in self.current_population:
                try:
                    fit = self.toolbox.evaluate(ind)
                except Exception as eval_e:
                    logger.error("Error evaluating individual %s: %s", ind, eval_e)
                    fit = 1e6  # Use a penalty value
                ind.fitness.values = (fit,)

            try:
                best_individual = tools.selBest(self.current_population, 1)[0]
                best_fitness = best_individual.fitness.values[0]
            except Exception as sel_e:
                logger.error("Generation %d: Error selecting best individual: %s", generation, sel_e)
                break

            logger.info("Generation %d: Best fitness after evaluation: %s", generation, best_fitness)

            # adjust dynamic probabilities based on fitness improvement
            if best_fitness < previous_best:
                # improvement: reduce probabilities (with saturation limits)
                self.crossover_probability = max(self.min_crossover_probability, self.crossover_probability - 0.05)
                self.mutation_probability = max(self.min_mutation_probability, self.mutation_probability - 0.05)
                logger.info(f"Generation {generation}: Fitness improved. Decreasing crossover_probability to "
                            f"{self.crossover_probability} and mutation_probability to {self.mutation_probability}")
            else:
                # no improvement or worse: increase probabilities (up to a maximum)
                self.crossover_probability = min(self.max_crossover_probability, self.crossover_probability + 0.05)
                self.mutation_probability = min(self.max_mutation_probability, self.mutation_probability + 0.05)
                logger.info(f"Generation {generation}: No improvement. Increasing crossover_probability to "
                            f"{self.crossover_probability} and mutation_probability to {self.mutation_probability}")

            previous_best = best_fitness

            if best_fitness < self.best_fitness:
                self.best_fitness = best_fitness
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            logger.info("Generation %d: Stagnation counter = %d", generation, self.stagnation_counter)
            if self.stagnation_counter >= self.stagnation_limit:
                logger.info("Generation %d: Stagnation limit reached. Stopping evolution.", generation)
                break

            # apply genetic operators using dynamic probabilities
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
                        if random.random() < self.crossover_probability:  # dynamic crossover probability
                            partner = random.choice(self.current_population)
                            self.toolbox.mate(mutant, self.toolbox.clone(partner))
                            logger.info("Generation %d: Crossover performed.", generation)
                        if random.random() < self.mutation_probability:  # dynamic mutation probability
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

        for individual in self.current_population:
            if not individual.fitness.values:
                individual.fitness.values = (1e6,)
        sorted_population = sorted(self.current_population, key=lambda ind: individual.fitness.values[0])
        return sorted_population[:k]

    def get_number_of_training_nodes(self):
        return self.number_of_training_nodes

    def save_population_to_json(self):
        """
        Save the current population to a json file.

        Each individual is stored as a dictionary with:
            - "chromosome": the list of hyperparameters values
            - "fitness": the fitness values (if available) or None.
        """
        if os.path.exists(self.genetic_population_file_path):
            os.remove(self.genetic_population_file_path)
            logger.info(f"Existing population file '{self.genetic_population_file_path}' deleted.")

        population_data = []
        for individual in self.current_population:
            # convert the individual (a list) and include fitness values if they exist.
            # in DEAP fitness.values is usually a tuple
            fitness = list(individual.fitness.values) if hasattr(individual, "fitness") and individual.fitness.values \
                else None
            individual_data = {
                "chromosome": list(individual),
                "fitness": fitness
            }
            population_data.append(individual_data)

        with open(self.genetic_population_file_path, "w") as f:
            json.dump(population_data, f, indent=2)
        logger.info(f"Saved population of {len(self.current_population)} individuals to "
                    f"{self.genetic_population_file_path}")

    def load_population_from_json(self):
        """
        Load the population from a JSON file.
        Each individual is expected to be a dictionary with keys:
            - "chromosome": a list of values
            - "fitness": a list (or None) representing the individual's fitness.
        If the file does not exist, a new population is generated using the toolbox.
        :return: list: a population (list of individuals) as used by DEAP
        """
        if os.path.exists(self.genetic_population_file_path):
            logger.info(f"Loading population from {self.genetic_population_file_path}")
            with open(self.genetic_population_file_path, "r") as f:
                population_data = json.load(f)

            population = []
            for individual_data in population_data:
                # create a new individual via the toolbox
                individual = self.toolbox.individual()
                # replace its content with the saved chromosome
                individual[:] = individual_data.get("chromosome", [])
                # set its fitness values if present
                fitness = individual_data.get("fitness")
                if fitness is not None:
                    individual.fitness.values = tuple(fitness)
                population.append(individual)
            logger.info(f"Loaded population of {len(population)} individuals from {self.genetic_population_file_path}")
            self.current_population = population
        else:
            logger.info(f"No saved population found at {self.genetic_population_file_path}. "
                        f"Initializing new population.")
            self.current_population = self.toolbox.population(n=self.population_size)
            logger.info("Initialized population with %d individuals", len(self.current_population))

