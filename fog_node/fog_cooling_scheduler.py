import math
import threading
import time

from enum import Enum
from fed_node.node_state import NodeState
from monitoring_thread import MonitoringThread
from typing import Callable, Any


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
        self.cooling_strategy = cooling_strategy
        self.stopping_score = 0

        # for boltzmann cooling strategy
        self.step = 1

        # for exponential cooling strategy
        self.exponential_coefficient = 0.95

        self.monitoring_thread = MonitoringThread(target=self.is_time_to_send_fog_model_to_cloud, sleep_time=1)
        self.send_fog_model_to_cloud = target

    def _cooling_process(self) -> None:
        """
        The cooling process that runs periodically in a separate thread
        """
        while self.is_cooling_operational:
            with self.lock:
                if self.temperature > self.cooling_threshold:
                    if self.cooling_strategy == CoolingStrategy.BOLTZMANN:
                        self.temperature /= math.log(self.step + 1)
                    elif self.cooling_strategy == CoolingStrategy.EXPONENTIAL:
                        self.temperature *= self.exponential_coefficient
                    else:
                        raise ValueError("Invalid cooling strategy. Use 'boltzmann' or 'exponential'.")

                    print(f"Cooling... Current temperature: {self.temperature:.4f}")
                    self.step += 1
                else:
                    self.is_cooling_operational = False
                    print("Cooling process has reached the cooldown threshold. Stopping...")
            # fixed-rated scheduling (2 seconds)
            time.sleep(2)

    def start_cooling(self):
        """
        Start the cooling process in a separate thread
        """
        with self.lock:
            self.is_cooling_operational = True
            self.temperature = self.initial_temperature
            self.step = 1
            print(f"Cooling process started with {self.cooling_strategy} strategy.")

        self.cooling_thread = threading.Thread(target=self._cooling_process, daemon=True)
        self.cooling_thread.start()
        self.monitoring_thread.start()

    def stop_cooling(self):
        """
        Stop the cooling process
        """
        with self.lock:
            self.is_cooling_operational = False

        if self.cooling_thread and self.cooling_thread.is_alive():
            self.cooling_thread.join()
        print("Cooling process stopped.")

    def is_cloud_cooling_operational(self) -> bool:
        return self.is_cooling_operational

    def increment_stopping_score(self) -> None:
        self.stopping_score += 1

    def has_reached_stopping_condition_for_cooler(self):
        if self.stopping_score == sum(1 for node in NodeState.get_current_node().child_nodes if
                                      not node.is_evaluation_node):
            self.stop_cooling()
            self.stopping_score = 0

    def is_time_to_send_fog_model_to_cloud(self):
        if self.has_reached_stopping_condition_for_cooler() or not self.is_cooling_operational:
            self.send_fog_model_to_cloud()
            self.monitoring_thread.stop()
