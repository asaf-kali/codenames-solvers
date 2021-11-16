import logging
import sys
from logging import Filter
from logging.config import dictConfig

log = logging.getLogger(__name__)
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(message)s",
        },
        "debug": {
            "format": "[%(asctime)s.%(msecs)03d] [%(levelname)-.4s]: %(message)s @@@ "
            "[%(threadName)s] [%(name)s:%(lineno)s]",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "filters": {
        "std_filter": {"()": "codenames.utils.LevelRangeFilter", "high": logging.WARNING},
        "err_filter": {"()": "codenames.utils.LevelRangeFilter", "low": logging.WARNING},
    },
    "handlers": {
        "console_out": {
            "class": "logging.StreamHandler",
            "filters": ["std_filter"],
            "formatter": "simple",
            "stream": sys.stdout,
        },
        "console_err": {
            "class": "logging.StreamHandler",
            "filters": ["err_filter"],
            "formatter": "debug",
            "stream": sys.stderr,
        },
    },
    "root": {"handlers": ["console_out", "console_err"], "level": "DEBUG"},
    "loggers": {
        "selenium": {"level": "INFO"},
        "urllib3": {"level": "INFO"},
        "codenames": {"level": "DEBUG"},
        # "codenames.online": {"level": "DEBUG"},
        # "codenames.solvers.naive": {"level": "INFO"},
    },
}


class LevelRangeFilter(Filter):
    def __init__(self, low=0, high=100):
        Filter.__init__(self)
        self.low = low
        self.high = high

    def filter(self, record):
        if self.low <= record.levelno < self.high:
            return True
        return False


def configure_logging():
    dictConfig(LOGGING_CONFIG)
    log.debug("Logging configured")


def wrap(o: object) -> str:
    return f"[{o}]"
