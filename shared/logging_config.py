import logging
import sys

# define a custom logging level "TEST" with a numeric value
TEST_LEVEL_NUM = 15
logging.addLevelName(TEST_LEVEL_NUM, "TEST")


# add a new method to the Logger class for the TEST level
def test(self, message, *args, **kwargs):
    if self.isEnabledFor(TEST_LEVEL_NUM):
        self._log(TEST_LEVEL_NUM, message, args, **kwargs)


logging.Logger.test = test


# create a custom formatter that colors TEST messages
class CustomFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLOR_CODES = {
        'TEST': "\033[35m",  # Magenta for TEST messages
    }

    def format(self, record):
        # if the record is of level TEST, modify levelname with color and custom text
        if record.levelname == "TEST":
            color = self.COLOR_CODES.get("TEST", "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# set up the logger as usual
logger = logging.getLogger("federated_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = CustomFormatter(
    '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False
