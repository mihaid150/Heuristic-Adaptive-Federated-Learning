# fog_node/genetic/genetic_engine.py

import random
import math
import os
import base64
import time
from typing import Any, Optional

import psutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from deap import base, tools
import json
from scipy.constants import Boltzmann
from shared.fed_node.node_state import NodeState
from shared.logging_config import logger
from fog_node.fog_resources_paths import FogResourcesPaths
from shared.shared_resources_paths import SharedResourcesPaths
from shared.fed_node.fed_node import MessageScope
from shared.utils import metric_weights, reinitialize_and_set_parent
from shared.fed_node.fed_node import FedNode


class FitnessMulti(base.Fitness):
    # Both performance score and resource load are minimized.
    weights = (-1.0, -1.0)


class Individual(list):
    """
    Represents an individual in the genetic algorithm with predefined attributes.
    """

    def __init__(self, learning_rate, batch_size, epochs, patience, fine_tune_layers):
        super().__init__([learning_rate, batch_size, epochs, patience, fine_tune_layers])
        self.fitness = FitnessMulti()

    @staticmethod
    def random_individual(learning_rate_bound, batch_size_bound, epochs_bound, patience_bound, fine_tune_layers_bound):
        """
        Creates a random individual with attributes initialized within their respective ranges.
        """
        learning_rate = random.randint(learning_rate_bound[0], learning_rate_bound[1])
        batch_size = random.randint(batch_size_bound[0], batch_size_bound[1])
        epochs = random.randint(epochs_bound[0], epochs_bound[1])
        patience = random.randint(patience_bound[0], patience_bound[1])
        fine_tune_layers = random.randint(fine_tune_layers_bound[0], fine_tune_layers_bound[1])
        return Individual(learning_rate, batch_size, epochs, patience, fine_tune_layers)


def select_evaluation_node():
    """
    Select an evaluation node from the list of current node children.
    Priority:
        - If one or more nodes have never been used (timestamp is None), choose one randomly.
        - Otherwise, select the node with the oldest last_time_fitness_evaluation_performed_timestamp.
    Also, mark only the chosen node's is_evaluation_node flag as True and reset others.
    """
    # filter nodes that have never been used for evaluation
    children_nodes = NodeState.get_current_node().child_nodes
    never_used = [node for node in children_nodes if node.last_time_fitness_evaluation_performed_timestamp is None]

    if never_used:
        selected = random.choice(never_used)
    else:
        selected = min(children_nodes, key=lambda node: node.last_time_fitness_evaluation_performed_timestamp)

    for node in children_nodes:
        node.is_evaluation_node = (node == selected)

    return selected


def get_system_metrics():
    # Get CPU usage over 1 second interval.
    cpu_usage = psutil.cpu_percent(interval=1)

    # Get memory usage percentage.
    memory_usage = psutil.virtual_memory().percent

    # Get average system load over the last 1, 5, and 15 minutes (only available on Unix systems).
    load_avg = sum(psutil.getloadavg()) / 3

    # Compute a combined resource load metric.
    # You could change the weights based on what's most relevant in your context.
    resource_load = (cpu_usage + memory_usage + load_avg) / 3

    cpu_freq_obj = psutil.cpu_freq()
    if cpu_freq_obj is not None and cpu_freq_obj.max:
        normalized_cpu_freq = (cpu_freq_obj.current / cpu_freq_obj.max) * 100
    else:
        normalized_cpu_freq = 0

    # Additional metric: Average sensor temperature
    sensor_temps = psutil.sensors_temperatures(fahrenheit=False)
    all_temps = []
    for sensor_list in sensor_temps.values():
        for reading in sensor_list:
            all_temps.append(reading.current)
    if all_temps:
        average_sensor_temp = sum(all_temps) / len(all_temps)
    else:
        average_sensor_temp = 0

    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "load_avg": load_avg,
        "normalized_cpu_freq": normalized_cpu_freq,
        "average_sensor_temp": average_sensor_temp,
        "resource_load": resource_load
    }


# uses a single-objective minimization
class GeneticEngine:
    def __init__(self):
        """
        Initializes the Genetic Engine
        """
        # initialization with default values until update
        self.population_size = 6
        self.number_of_generations = 4
        self.stagnation_limit = 3

        self.toolbox: Any = base.Toolbox()
        self.best_fitness = float("inf")
        self.stagnation_counter = 0
        self.boltzmann_constant = Boltzmann
        self.additional_factor = 1e23
        self.current_population = None
        self.operating_data_date = []
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

        self.learning_rate_bound = (1, 100)
        self.batch_size_bound = (32, 64)
        self.epochs_bound = (10, 20)
        self.patience_bound = (5, 10)
        self.fine_tune_layers_bound = (1, 3)

        # DEAP statistics, logbook, and hall of fame setup
        # we use a statistics object over the fitness values (assuming one-element tuples)
        self.stats = tools.Statistics(self.safe_fitness)
        # logbook to keep track of statistics per generation
        self.logbook = tools.Logbook()
        self.DEFAULT_LOGBOOK_HEADER = ["gen", "nevals", "avg", "std", "min", "max", "genotypic_diversity",
                                       "phenotypic_diversity"]
        self.logbook.header = self.DEFAULT_LOGBOOK_HEADER
        # hall of fame to maintain best individuals
        self.hall_of_fame = tools.HallOfFame(1)

        self.cloud_node: Optional['FedNode'] = None

        self.evolution_system_metrics = []
        self.evolution_system_metrics_buffer = []

    def configure_training_parameters_bounds(self, lr_min, lr_max, bs_min, bs_max, ep_min, ep_max, pa_min, pa_max,
                                             ftl_min, ftl_max):
        self.learning_rate_bound = (lr_min, lr_max)
        self.batch_size_bound = (bs_min, bs_max)
        self.epochs_bound = (ep_min, ep_max)
        self.patience_bound = (pa_min, pa_max)
        self.fine_tune_layers_bound = (ftl_min, ftl_max)

        logger.info(f"Successfully set the training parameter bounds to: learning rate {self.learning_rate_bound}, "
                    f" batch size {self.batch_size_bound}, epochs {self.epochs_bound}, patience {self.patience_bound},"
                    f" fine tune layers {self.fine_tune_layers_bound}.")

    def get_current_training_parameter_bounds(self):
        return {
            "learning_rate_lower_bound": self.learning_rate_bound[0],
            "learning_rate_upper_bound": self.learning_rate_bound[1],
            "batch_size_lower_bound": self.batch_size_bound[0],
            "batch_size_upper_bound": self.batch_size_bound[1],
            "epochs_lower_bound": self.epochs_bound[0],
            "epochs_upper_bound": self.epochs_bound[1],
            "patience_lower_bound": self.patience_bound[0],
            "patience_upper_bound": self.patience_bound[1],
            "fine_tune_layers_lower_bound": self.fine_tune_layers_bound[0],
            "fine_tune_layers_upper_bound": self.fine_tune_layers_bound[1]
        }

    def set_genetic_engine_parameters(self, population_size, number_of_generations, stagnation_limit):
        if population_size is not None:
            self.population_size = population_size

        if number_of_generations is not None:
            self.number_of_generations = number_of_generations

        if stagnation_limit is not None:
            self.stagnation_limit = stagnation_limit

        self.adjust_population_size()

    def get_genetic_engine_parameters(self):
        return {
            "population_size": self.population_size,
            "number_of_generations": self.number_of_generations,
            "stagnation_limit": self.stagnation_limit,
        }

    def set_cloud_node(self, cloud_node: FedNode):
        self.cloud_node = cloud_node

    def adjust_population_size(self):
        """
        Adjust the current population to mathc the updated population_size.
        If the new size is smaller than the current population, only the fittest individuals are retained
        If the new size is larger, new individuals are added.
        """

        if self.current_population is None:
            if os.path.exists(self.genetic_population_file_path):
                self.load_population_from_json()
            else:
                return

        current_size = len(self.current_population)
        if self.population_size < current_size:
            # sort individual by fitness and for the ones with invalid fitness use infinity
            self.current_population.sort(key=lambda ind: ind.fitness.values[0] if ind.fitness.values else float("inf"))
            self.current_population = self.current_population[:self.population_size]
            logger.info(f"Population truncated to {self.population_size} individuals.")
        elif self.population_size > current_size:
            # add new individuals until population size is met
            num_to_add = self.population_size - current_size
            for _ in range(num_to_add):
                new_individual = self.toolbox.individual()
                self.current_population.append(new_individual)
            logger.info(f"{num_to_add} new individuals added to reach a total of {self.population_size}.")
        else:
            logger.info("Population size matches the new parameters, thus no adjustment needed.")

    def get_cloud_temperature(self):
        url = f"http://{self.cloud_node.ip_address}:{self.cloud_node.port}/cloud/get-cloud-temperature"
        logger.info(f"Sending HTTP POST request to {url} for cloud temperature request.")
        try:
            r = requests.post(url, timeout=None)
            logger.info(f"Received HTTP response from {url}: status code {r.status_code}")
            if 200 <= r.status_code < 300:
                logger.info(f"Response from {url} accepted.")
                temp = r.json().get("cloud_temperature")
                if temp is not None:
                    return temp
                else:
                    return random.uniform(1, 100)
        except Exception as e:
            logger.info(f"HTTP request error to {url}: {e}")
            return random.uniform(1, 100)

    def set_operating_data_date(self, dates):
        self.operating_data_date = dates

    def clear_operating_data_date(self):
        self.operating_data_date = []

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
            return Individual.random_individual(self.learning_rate_bound, self.batch_size_bound, self.epochs_bound,
                                                self.patience_bound, self.fine_tune_layers_bound)

        # register individual and population creation
        self.toolbox.register("individual", create_individual)

        # register population creation
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # register evaluate function creation
        self.toolbox.register("evaluate", self.fitness_function)

        # genetic operators
        self.toolbox.register("mate", tools.cxOnePoint)  # crossover
        self.toolbox.register("mutate", tools.mutUniformInt, low=[self.learning_rate_bound[0], self.batch_size_bound[0],
                                                                  self.epochs_bound[0], self.patience_bound[0],
                                                                  self.fine_tune_layers_bound[0]], up=[
            self.learning_rate_bound[1], self.batch_size_bound[1],
            self.epochs_bound[1], self.patience_bound[1],
            self.fine_tune_layers_bound[1]], indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # selection

        self.stats.register("avg", lambda fits: sum(fits) / len(fits))
        self.stats.register("std", lambda fits: math.sqrt(sum((x - sum(fits) / len(fits)) ** 2 for x in fits) /
                                                          len(fits)) if fits else 0)
        self.stats.register("min", min)
        self.stats.register("max", max)

        self.logbook.header = ["gen", "nevals", "avg", "std", "min", "max", "genotypic_diversity",
                               "phenotypic_diversity"]

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
            "fine_tune_layers": fine_tune_layers,
            "scope": MessageScope.TRAINING.value
        }
        logger.info(f"Payload template built with keys: {list(payload_template.keys())}")

        # Get all evaluation nodes.
        evaluation_nodes = [child for child in NodeState.get_current_node().child_nodes if child.is_evaluation_node]
        logger.info(f"Found {len(evaluation_nodes)} evaluation nodes.")

        def send_request(node):
            payload = payload_template.copy()
            payload["child_id"] = node.id
            url = f"http://{node.ip_address}:{node.port}/edge/execute-model-evaluation"
            logger.info(f"Sending HTTP POST request to {url} with child: {node.name}.")
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

        max_trials = 3
        # Create a dictionary to track the number of trials per node.
        trial_counts = {node: 0 for node in evaluation_nodes}
        valid_responses = []

        # Continue looping until we have at least one valid response
        # or all nodes have exhausted their maximum trials.
        while not valid_responses and any(trial < max_trials for trial in trial_counts.values()):
            # Only try nodes that have remaining trials.
            nodes_to_try = [node for node in evaluation_nodes if trial_counts[node] < max_trials]

            if not nodes_to_try:
                break

            with ThreadPoolExecutor(max_workers=len(nodes_to_try)) as executor:
                futures = {executor.submit(send_request, node): node for node in nodes_to_try}
                logger.info("Submitted requests to evaluation nodes; waiting for responses...")

                for future in as_completed(futures):
                    node = futures[future]
                    try:
                        res = future.result()
                        if res is not None:
                            logger.info("Received valid response from node %s", node.name)
                            valid_responses.append(res)
                        else:
                            trial_counts[node] += 1
                            logger.error("No valid response from node %s (trial %d)", node.id, trial_counts[node])
                            if trial_counts[node] < max_trials:
                                logger.info("Waiting 5 minutes before retrying node %s.", node.name)
                                time.sleep(300)
                                reinitialize_and_set_parent(node, NodeState.get_current_node())
                                node.last_time_fitness_evaluation_performed_timestamp = time.time_ns()
                                # Select a (potentially new) evaluation node for retrying.
                                new_node = select_evaluation_node()
                                logger.info("Retrying request using new evaluation node %s.", new_node.name)
                                # Replace the old node with the new one and preserve the trial count.
                                trial_counts[new_node] = trial_counts.pop(node)
                                # Also update evaluation_nodes list so the new node is used in future iterations.
                                evaluation_nodes[evaluation_nodes.index(node)] = new_node
                    except Exception as e:
                        trial_counts[node] += 1
                        logger.error("Exception while processing response from node %s (trial %d): %s", node.name,
                                     trial_counts[node], e)

        if not valid_responses:
            raise Exception("Maximum retries exceeded for all nodes with no valid response.")

        def compute_weighted_score(metrics, local_weights):
            return sum(local_weights[metric] * metrics[metric] for metric in local_weights)

        scores = []
        resource_loads = []
        for resp in valid_responses:
            try:
                score_before = compute_weighted_score(resp["metrics"]["before_training"], metric_weights)
                score_after = compute_weighted_score(resp["metrics"]["after_training"], metric_weights)
                score = min(score_before, score_after)
                scores.append(score)
                logger.info("Computed score for response: before=%s, after=%s, chosen=%s", score_before, score_after,
                            score)
            except Exception as e:
                logger.error("Error computing score for a response: %s", e)
                scores.append(1e6)  # Penalty score

            try:
                # Extract resource load from system metrics provided in the response.
                system_metrics = resp.get("system_metrics")
                self.evolution_system_metrics_buffer.append(system_metrics)
                logger.info(f"Received system metrics: {system_metrics}")
                if system_metrics and "resource_load" in system_metrics:
                    resource_loads.append(system_metrics["resource_load"])
                else:
                    # Fallback to an external measurement if not provided.
                    fallback_metrics = 1e6
                    logger.info(f"Using fallback system metrics: {fallback_metrics}")
                    resource_loads.append(fallback_metrics)
            except Exception as e:
                logger.error("Error retrieving system metrics from response: %s", e)
                resource_loads.append(1e6)

        if scores:
            final_score = sum(scores) / len(scores)
        else:
            final_score = 1e6

        if resource_loads:
            avg_resource_load = sum(resource_loads) / len(resource_loads)
        else:
            avg_resource_load = 1e6

        logger.info("Final multi-objective fitness computed: performance score = %s, resource load = %s", final_score,
                    avg_resource_load)
        return final_score, avg_resource_load

    def evolve(self):
        self.logbook = tools.Logbook()
        self.logbook.header = self.DEFAULT_LOGBOOK_HEADER
        self.stagnation_counter = 0
        self.best_fitness = float("inf")

        logger.info("Starting evolution with population size: %d, generations: %d",
                    self.population_size, self.number_of_generations)
        if not self.toolbox:
            raise ValueError("Toolbox is not set up. Call setup() before evolve().")
        if self.current_population is None:
            self.load_population_from_json()

        logger.info("Initial population size: %d", len(self.current_population))
        previous_best = self.best_fitness

        for generation in range(self.number_of_generations):
            self.evolution_system_metrics_buffer = []
            logger.info("Generation %d: Current population size: %d", generation, len(self.current_population))
            selected_node = select_evaluation_node()
            logger.info("Generation %d: Selected evaluation node %s.", generation, selected_node.name)

            # Evaluate each individual.
            for idx, ind in enumerate(self.current_population):
                try:
                    ind.fitness.values = self.toolbox.evaluate(ind)
                except Exception as eval_e:
                    logger.error("Error evaluating individual %s: %s", ind, eval_e)
                    ind.fitness.values = (1e6, 1e6)
                if not ind.fitness.values or not all(math.isfinite(val) for val in ind.fitness.values):
                    logger.error("Evaluation failed for individual %d: %s", idx, ind.fitness.values)

            # Use NSGA-II for multi-objective selection.
            # The entire population is ranked according to Pareto dominance.
            self.current_population = tools.selNSGA2(self.current_population, k=len(self.current_population))
            # For logging or early stopping, choose a single "best" individual by an aggregate measure.
            best_individual = min(self.current_population, key=lambda lind: sum(lind.fitness.values))
            best_fitness = sum(best_individual.fitness.values)

            logger.info("Generation %d: Best fitness after evaluation: %s", generation, best_fitness)
            pop_fitness = [ind.fitness.values[0] for ind in self.current_population if ind.fitness.values]
            logger.info("Generation %d: Population fitness values: %s", generation, pop_fitness)

            # Adjust dynamic probabilities based on fitness improvement.
            if best_fitness < previous_best:
                self.crossover_probability = max(self.min_crossover_probability, self.crossover_probability - 0.05)
                self.mutation_probability = max(self.min_mutation_probability, self.mutation_probability - 0.05)
                logger.info(
                    "Generation %d: Fitness improved. Decreasing crossover_probability to %s and mutation_probability "
                    "to %s",
                    generation, self.crossover_probability, self.mutation_probability)
            else:
                self.crossover_probability = min(self.max_crossover_probability, self.crossover_probability + 0.05)
                self.mutation_probability = min(self.max_mutation_probability, self.mutation_probability + 0.05)
                logger.info(
                    "Generation %d: No improvement. Increasing crossover_probability to %s and mutation_probability to "
                    "%s",
                    generation, self.crossover_probability, self.mutation_probability)

            previous_best = best_fitness

            if best_fitness < self.best_fitness:
                self.best_fitness = best_fitness
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            logger.info("Generation %d: Stagnation counter = %d", generation, self.stagnation_counter)
            if self.stagnation_counter >= self.stagnation_limit:
                logger.info("Generation %d: Stagnation limit reached with population size %d. Stopping evolution.",
                            generation, len(self.current_population))
                break

            # Apply genetic operators with immediate re-evaluation for modified individuals.
            offspring = []
            cloud_temperature = self.get_cloud_temperature()
            logger.info("Generation %d: Starting genetic operations with cloud_temperature = %s", generation,
                        cloud_temperature)
            for idx, individual in enumerate(self.current_population):
                try:
                    if self.should_skip_update(individual.fitness.values[0], cloud_temperature):
                        # Clone without change; preserve fitness.
                        clone_individual = self.toolbox.clone(individual)
                        offspring.append(clone_individual)
                        logger.info("Generation %d: Individual %d skipped update (cloned) with fitness preserved.",
                                    generation, idx)
                    else:
                        # Clone and then perform genetic operations.
                        mutant = self.toolbox.clone(individual)
                        modified = False
                        if random.random() < self.crossover_probability:
                            partner = random.choice(self.current_population)
                            self.toolbox.mate(mutant, self.toolbox.clone(partner))
                            modified = True
                            logger.info("Generation %d: Crossover performed for individual %d.", generation, idx)
                        if random.random() < self.mutation_probability:
                            self.toolbox.mutate(mutant)
                            modified = True
                            logger.info("Generation %d: Mutation performed for individual %d.", generation, idx)
                        if modified:
                            try:
                                new_fit = self.toolbox.evaluate(mutant)
                            except Exception as e:
                                logger.error("Generation %d: Error re-evaluating mutated individual %d: %s", generation,
                                             idx, e)
                                new_fit = (1e6, 1e6)
                            mutant.fitness.values = new_fit
                            logger.info("Generation %d: Individual %d re-evaluated; new fitness: %s", generation, idx,
                                        new_fit)
                        else:
                            logger.info(
                                "Generation %d: No genetic operator applied for individual %d; fitness retained.",
                                generation, idx)
                        offspring.append(mutant)
                except Exception as op_e:
                    logger.error("Generation %d: Error processing genetic operator for individual %d: %s", generation,
                                 idx, op_e)
            logger.info("Generation %d: Genetic operations complete. Offspring count = %d", generation, len(offspring))
            self.current_population[:] = offspring
            logger.info("Generation %d: End of generation. New population size: %d", generation,
                        len(self.current_population))
            selected_node.last_time_fitness_evaluation_performed_timestamp = time.time_ns()

            # Double-check that all individuals have valid (finite) fitness values.
            for idx, ind in enumerate(self.current_population):
                if not ind.fitness.values or not all(math.isfinite(val) for val in ind.fitness.values):
                    try:
                        fit = self.toolbox.evaluate(ind)
                    except Exception as e:
                        logger.error("Re-evaluation error for individual %d: %s", idx, e)
                        fit = (1e6, 1e6)
                    ind.fitness.values = fit
                    logger.info("Individual %d re-evaluated during final check; new fitness: %s", idx,
                                ind.fitness.values)

            # Now compile and record statistics.
            record = self.stats.compile(self.current_population)
            record["genotypic_diversity"] = self.compute_genotypic_diversity()
            record["phenotypic_diversity"] = self.compute_phenotypic_diversity()
            self.logbook.record(gen=generation, nevals=len(self.current_population), **record)
            logger.info("Generation %d: Logbook record: %s", generation, self.logbook.stream)
            self.hall_of_fame.update(self.current_population)

            average_cpu_usage = (sum(sys_metrics["cpu_usage"] for sys_metrics in self.evolution_system_metrics_buffer) /
                                 len(self.evolution_system_metrics_buffer))
            average_memory_usage = (sum(sys_metrics["memory_usage"] for sys_metrics in
                                        self.evolution_system_metrics_buffer) /
                                    len(self.evolution_system_metrics_buffer))
            average_load_avg = (sum(sys_metrics["load_avg"] for sys_metrics in self.evolution_system_metrics_buffer) /
                                len(self.evolution_system_metrics_buffer))
            average_resource_load = (sum(sys_metrics["resource_load"] for sys_metrics in
                                         self.evolution_system_metrics_buffer) /
                                     len(self.evolution_system_metrics_buffer))
            average_normalized_cpu_freq = (sum(sys_metrics["normalized_cpu_freq"] for sys_metrics in
                                               self.evolution_system_metrics_buffer) /
                                           len(self.evolution_system_metrics_buffer))
            average_average_sensor_temp = (sum(sys_metrics["average_sensor_temp"] for sys_metrics in
                                               self.evolution_system_metrics_buffer) /
                                           len(self.evolution_system_metrics_buffer))

            generation_system_metrics_average = {
                "gen": generation,
                "cpu_usage": average_cpu_usage,
                "memory_usage": average_memory_usage,
                "load_avg": average_load_avg,
                "normalized_cpu_freq": average_normalized_cpu_freq,
                "average_sensor_temp": average_average_sensor_temp,
                "resource_load": average_resource_load
            }

            self.evolution_system_metrics.append(generation_system_metrics_average)

        logger.info("Evolution complete! Best individual (hall of fame): %s with fitness %s",
                    self.hall_of_fame[0], self.hall_of_fame[0].fitness.values[0])

    def get_top_k_individuals(self, k):
        logger.info(f"Requested individuals: {k}, Current Population Size {len(self.current_population)}.")
        if self.current_population is None:
            return ValueError("Population is empty. Ensure evolution has been run before calling this method.")
        if k > len(self.current_population):
            raise ValueError(f"Requested top {k} individuals, but population size is {len(self.current_population)}.")
        for individual in self.current_population:
            if not individual.fitness.values:
                individual.fitness.values = (1e6, 1e6)
        sorted_population = sorted(self.current_population, key=lambda ind: sum(ind.fitness.values))
        return sorted_population[:k]

    def save_population_to_json(self):
        if os.path.exists(self.genetic_population_file_path):
            os.remove(self.genetic_population_file_path)
            logger.info(f"Existing population file '{self.genetic_population_file_path}' deleted.")
        population_data = []
        for individual in self.current_population:
            fitness = list(individual.fitness.values) if hasattr(individual,
                                                                 "fitness") and individual.fitness.values else None
            individual_data = {
                "chromosome": list(individual),
                "fitness": fitness
            }
            population_data.append(individual_data)
        header = self.logbook.header if self.logbook.header else self.DEFAULT_LOGBOOK_HEADER
        logbook_records = []
        for record in self.logbook:
            rec_list = [record.get(key) for key in header]
            logbook_records.append(rec_list)
        logbook_data = {"header": header, "records": logbook_records}
        data_to_save = {"population": population_data, "logbook": logbook_data}
        with open(self.genetic_population_file_path, "w") as f:
            json.dump(data_to_save, f, indent=2)
        logger.info(
            f"Saved population and logbook of {len(self.current_population)} individuals to "
            f"{self.genetic_population_file_path}")

    def load_population_from_json(self):
        if os.path.exists(self.genetic_population_file_path):
            logger.info(f"Loading population and logbook from {self.genetic_population_file_path}")
            with open(self.genetic_population_file_path, "r") as f:
                data = json.load(f)
            population_data = data.get("population", [])
            logbook_data = data.get("logbook", None)
            population = []
            for individual_data in population_data:
                individual = self.toolbox.individual()
                individual[:] = individual_data.get("chromosome", [])
                fitness = individual_data.get("fitness")
                if fitness is not None:
                    individual.fitness.values = tuple(fitness)
                population.append(individual)
            logger.info(f"Loaded population of {len(population)} individuals from {self.genetic_population_file_path}")
            self.current_population = population

            if logbook_data is not None:
                logger.info(f"Logbook data type: {type(logbook_data)}")
                self.logbook = tools.Logbook()
                if isinstance(logbook_data, dict):
                    header = logbook_data.get("header", [])
                    records = logbook_data.get("records", [])
                elif isinstance(logbook_data, list):
                    logger.warning(
                        "Logbook data is a list; expected dict with 'header' and 'records'. Using default header.")
                    header = list(logbook_data[0].keys()) if logbook_data and isinstance(logbook_data[0], dict) else []
                    records = logbook_data
                else:
                    logger.error(f"Unexpected logbook data type: {type(logbook_data)}. Initializing empty logbook.")
                    header = []
                    records = []
                filtered_records = []
                for record in records:
                    if all(not (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v))) for v in record):
                        filtered_records.append(record)
                    else:
                        logger.warning("Removed logbook record with invalid value: %s", record)
                self.logbook.header = header
                logger.info(f"Logbook header loaded: {header}")
                for record in filtered_records:
                    if isinstance(record, dict):
                        self.logbook.record(**record)
                    elif isinstance(record, list):
                        rec_dict = dict(zip(header, record))
                        self.logbook.record(**rec_dict)
                    else:
                        logger.error("Unexpected logbook record format: %s", record)
                logger.info(
                    f"Loaded logbook with {len(filtered_records)} valid records from "
                    f"{self.genetic_population_file_path}")
            else:
                logger.info("No logbook data found in saved file. Initializing empty logbook.")
                self.logbook = tools.Logbook()
        else:
            logger.info(
                f"No saved population found at {self.genetic_population_file_path}. Initializing new population.")
            self.current_population = self.toolbox.population(n=self.population_size)
            logger.info("Initialized population with %d individuals", len(self.current_population))

    def compute_genotypic_diversity(self):
        distances = []
        for i in range(len(self.current_population)):
            for j in range(i + 1, len(self.current_population)):
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.current_population[i], self.current_population[j])))
                distances.append(d)
        average_distance = sum(distances) / len(distances) if distances else 0
        return average_distance

    def compute_phenotypic_diversity(self):
        fitness_values = [ind.fitness.values[0] for ind in self.current_population if ind.fitness.values]
        if not fitness_values:
            return 0
        mean_fitness = sum(fitness_values) / len(fitness_values)
        variance = sum((x - mean_fitness) ** 2 for x in fitness_values) / len(fitness_values)
        return math.sqrt(variance)

    def safe_fitness(self, ind):
        if not ind.fitness.values or not math.isfinite(ind.fitness.values[0]):
            try:
                new_fit = self.toolbox.evaluate(ind)
            except Exception as e:
                logger.error("Error re-evaluating fitness for individual %s: %s", ind, e)
                new_fit = (1e6, 1e6)
            ind.fitness.values = new_fit
        return sum(ind.fitness.values)
