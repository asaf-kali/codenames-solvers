from logging.config import dictConfig
from typing import Optional

from the_spymaster_util.logger import get_dict_config


def configure_logging(
    formatter: Optional[str] = None, level: Optional[str] = None, mute_solvers: bool = False, mute_online: bool = True
):
    # handlers = {
    #     "file": {
    #         "class": "logging.FileHandler",
    #         "filename": "run.log",
    #         "formatter": "debug",
    #     },
    # }
    loggers = {
        "selenium": {"level": "INFO"},
        "urllib3": {"level": "INFO"},
        "matplotlib.font_manager": {"propagate": False},
        "the_spymaster_util.async_task_manager": {"level": "INFO"},
        "openai": {"level": "INFO"},
    }
    dict_config = get_dict_config(
        std_formatter=formatter,
        root_log_level=level,
        # extra_handlers=handlers,
        extra_loggers=loggers,
    )
    # dict_config["root"]["handlers"].append("file")

    # if mute_solvers:
    # dict_config["loggers"]["solvers"] = {"handlers": ["file"], "propagate": False}  # type: ignore
    # if mute_online:
    # dict_config["loggers"]["codenames.online"] = {"handlers": ["file"], "propagate": False}  # type: ignore
    dictConfig(dict_config)
    print("Logging configured.")
