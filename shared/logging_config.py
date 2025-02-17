import logging
import sys

logger = logging.getLogger("federated_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False  # prevent messages from bubbling up to the root logger
