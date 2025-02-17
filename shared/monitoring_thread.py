import threading
import time
from typing import Callable, Any
from shared.logging_config import logger


class MonitoringThread(threading.Thread):
    def __init__(self, target: Callable[..., Any], sleep_time: float = 1, *args, **kwargs):
        """
        Initialize the thread with a target function to run in a loop.

        :param target: The function to execute repeatedly.
        :param args: Positional arguments for the target function.
        :param kwargs: Keyword arguments for the target function.
        """
        super().__init__()
        # event to signal when to stop
        self._stop_event = threading.Event()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.delay = sleep_time

    def run(self):
        # check the stop signal
        logger.info("Monitoring thread started.")
        while not self._stop_event.is_set():
            self.target(*self.args, **self.kwargs)  # call the target function
            time.sleep(self.delay)
        logger.info("Monitoring thread stopping...")

    def stop(self):
        """
        Signal the thread to stop.
        """
        self._stop_event.set()
