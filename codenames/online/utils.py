import logging
import time
from time import sleep
from typing import Callable, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


class PollingTimeout(Exception):
    pass


def poll_not_none(fn: Callable[[], T], timeout_seconds: float = 5, interval_sleep_seconds: float = 0.2) -> T:
    start = time.time()
    while True:
        result = fn()
        if result is not None:
            return result
        log.debug("Result was none, sleeping...")
        now = time.time()
        passed = now - start
        if passed >= timeout_seconds:
            raise PollingTimeout()
        sleep(interval_sleep_seconds)


def poll_condition(test: Callable[[], bool], timeout_seconds: float = 5, interval_sleep_seconds: float = 0.2):
    start = time.time()
    while not test():
        log.debug("Test not passed, sleeping...")
        now = time.time()
        passed = now - start
        if passed >= timeout_seconds:
            raise PollingTimeout()
        sleep(interval_sleep_seconds)
