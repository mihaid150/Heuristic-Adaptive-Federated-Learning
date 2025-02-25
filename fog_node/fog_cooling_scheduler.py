import math
import threading
import time
from enum import Enum
from shared.fed_node.node_state import NodeState
from shared.monitoring_thread import MonitoringThread
from shared.logging_config import logger
from typing import Callable, Any
from shared.fed_node.fed_node import ModelScope


class CoolingStrategy(Enum):
    BOLTZMANN = 1
    EXPONENTIAL = 2


class FogCoolingScheduler:
    def __init__(self, target: Callable[..., Any], cooling_strategy: CoolingStrategy = CoolingStrategy.BOLTZMANN) -> \
            None:
        """
        Initialize the cooling scheduler with default parameters.
        :param cooling_strategy: 'boltzmann' or 'exponential'
        """
        self.initial_temperature = 100.0
        self.cooling_threshold = 0.01
        self.temperature = self.initial_temperature
        self.is_cooling_operational = False
        self.lock = threading.Lock()
        self.cooling_thread = None
        self.monitoring_thread = None
        self.cooling_strategy = cooling_strategy
        self.stopping_score = 0

        # For Boltzmann cooling strategy
        self.step = 1
        self.boltzmann_coefficient = 0.001  # Controls the cooling rate for Boltzmann

        # For exponential cooling strategy
        self.exponential_coefficient = 0.999  # Reduced cooling rate

        self.send_fog_model_to_cloud = target

    def _cooling_process(self) -> None:
        """
        The cooling process that runs periodically in a separate thread.
        """
        counter = 0
        while self.is_cooling_operational:
            with self.lock:
                if self.temperature > self.cooling_threshold:
                    if self.cooling_strategy == CoolingStrategy.BOLTZMANN:
                        self.temperature -= self.boltzmann_coefficient * self.temperature / math.log(self.step + 2)
                    elif self.cooling_strategy == CoolingStrategy.EXPONENTIAL:
                        self.temperature *= self.exponential_coefficient
                    else:
                        raise ValueError("Invalid cooling strategy. Use 'boltzmann' or 'exponential'.")

                    # Ensure temperature never exceeds the initial value
                    self.temperature = min(self.temperature, self.initial_temperature)
                    self.step += 1
                else:
                    self.is_cooling_operational = False
                    logger.info("Cooling process has reached the cooldown threshold. Stopping...")

            # Increase counter and log every 12 iterations (i.e., every 60 seconds)
            counter += 1
            if counter % 12 == 0:
                logger.info(f"Cooling... Current temperature: {self.temperature:.4f}")

            # Sleep for 5 seconds between iterations
            time.sleep(5)

    def start_cooling(self):
        """
        Start the cooling process in a separate thread
        """
        with self.lock:
            self.is_cooling_operational = True
            self.temperature = self.initial_temperature
            self.step = 1
            logger.info(f"Cooling process started with {self.cooling_strategy} strategy.")

        self.cooling_thread = threading.Thread(target=self._cooling_process, daemon=True)
        self.cooling_thread.start()
        self.monitoring_thread = MonitoringThread(target=self.is_time_to_send_fog_model_to_cloud, sleep_time=1)
        self.monitoring_thread.start()

    def stop_cooling(self):
        """
        Stop the cooling process
        """
        with self.lock:
            self.is_cooling_operational = False

        if self.cooling_thread and self.cooling_thread.is_alive():
            self.cooling_thread.join()
        logger.info("Cooling process stopped.")

    def is_fog_cooling_operational(self) -> bool:
        return self.is_cooling_operational

    def increment_stopping_score(self) -> None:
        self.stopping_score += 1

    def has_reached_stopping_condition_for_cooler(self) -> bool:
        if self.stopping_score == sum(1 for node in NodeState.get_current_node().child_nodes if
                                      not node.is_evaluation_node):
            self.stop_cooling()
            self.stopping_score = 0
            return True
        return False

    def is_time_to_send_fog_model_to_cloud(self):
        if self.has_reached_stopping_condition_for_cooler() or not self.is_cooling_operational:
            self.send_fog_model_to_cloud(ModelScope.TRAINING, None)
            self.monitoring_thread.stop()

    def reset(self):
        """
        Reset the scheduler's internal state so that a new cooling cycle can be started.
        """
        with self.lock:
            self.is_cooling_operational = False
            self.temperature = self.initial_temperature
            self.step = 1
            self.stopping_score = 0
            # Reset the monitoring thread by creating a new instance.
            self.monitoring_thread = MonitoringThread(
                target=self.is_time_to_send_fog_model_to_cloud,
                sleep_time=1
            )
        logger.info("FogCoolingScheduler has been reset.")
