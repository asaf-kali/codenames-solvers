import logging
import sys
from datetime import datetime
from logging import Filter, Formatter, Logger, LogRecord
from logging.config import dictConfig


class ExtraDataLogger(Logger):
    def _log(self, *args, **kwargs) -> None:
        extra = kwargs.get("extra")
        if extra is not None:
            kwargs["extra"] = {"extra": extra}
        super()._log(*args, **kwargs)  # noqa


class ExtraDataFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        extra = getattr(record, "extra", None)
        if extra:
            record.msg += f" {extra}"
        return super().format(record)


class LevelRangeFilter(Filter):
    def __init__(self, low=0, high=100):
        Filter.__init__(self)
        self.low = low
        self.high = high

    def filter(self, record):
        if self.low <= record.levelno < self.high:
            return True
        return False


logging.setLoggerClass(ExtraDataLogger)

log = logging.getLogger(__name__)


def configure_logging(formatter: str = None, level: str = None, mute_solvers: bool = False, mute_online: bool = True):
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"class": "codenames.utils.ExtraDataFormatter"},
            "debug": {
                "class": "codenames.utils.ExtraDataFormatter",
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
                "formatter": formatter or "simple",
                "stream": sys.stdout,
            },
            "console_err": {
                "class": "logging.StreamHandler",
                "filters": ["err_filter"],
                "formatter": formatter or "debug",
                "stream": sys.stdout,
                # "stream": sys.stderr,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "run.log",
                "formatter": formatter or "debug",
            },
        },
        "root": {"handlers": ["console_out", "console_err", "file"], "level": level or "DEBUG"},
        "loggers": {
            "selenium": {"level": "INFO"},
            "urllib3": {"level": "INFO"},
            "matplotlib.font_manager": {"propagate": False},
            # "findfont": {"level": "error"},
        },
    }

    if mute_solvers:
        config["loggers"]["codenames.solvers"] = {"handlers": ["file"], "propagate": False}  # type: ignore
    if mute_online:
        config["loggers"]["codenames.online"] = {"handlers": ["file"], "propagate": False}  # type: ignore
    dictConfig(config)
    log.debug("Logging configured")


def wrap(o: object) -> str:
    return f"[{o}]"


RUN_ID = datetime.now().timestamp()
