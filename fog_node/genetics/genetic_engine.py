import random
import math
from deap import base, tools
from scipy.constants import Boltzmann


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
        self.evaluation_data_date = []

    @staticmethod
    def get_cloud_temperature():
        return random.uniform(1, 100)

    def set_evaluation_data_date(self, dates):
        self.evaluation_data_date = dates

    def clear_evaluation_data_date(self):
        self.evaluation_data_date = []

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

    @staticmethod
    def fitness_function(individual):
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

        # temporary simulate model evaluation
        fitness = (
                abs(learning_rate - 0.001)
                + abs(batch_size - 64)
                + abs(number_epochs - 50)
                + abs(patience - 10)
                + abs(fine_tune_layers - 5)
        )

        return fitness

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
