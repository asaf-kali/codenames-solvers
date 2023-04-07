import logging
from logging import Filter, Formatter, Logger, LogRecord
from logging.config import dictConfig
from typing import Optional

from the_spymaster_util.logger import get_dict_config


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


def configure_logging(
    formatter: Optional[str] = None, level: Optional[str] = None, mute_solvers: bool = False, mute_online: bool = True
):
    handlers = {
        "file": {
            "class": "logging.FileHandler",
            "filename": "run.log",
            "formatter": "debug",
        },
    }
    loggers = {
        "selenium": {"level": "INFO"},
        "urllib3": {"level": "INFO"},
        "matplotlib.font_manager": {"propagate": False},
        "the_spymaster_util.async_task_manager": {"level": "INFO"},
    }
    dict_config = get_dict_config(
        std_formatter=formatter,
        root_log_level=level,
        extra_handlers=handlers,
        extra_loggers=loggers,
    )
    dict_config["root"]["handlers"].append("file")

    if mute_solvers:
        dict_config["loggers"]["solvers"] = {"handlers": ["file"], "propagate": False}  # type: ignore
    if mute_online:
        dict_config["loggers"]["codenames.online"] = {"handlers": ["file"], "propagate": False}  # type: ignore
    dictConfig(dict_config)
    log.debug("Logging configured")
