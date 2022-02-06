import logging
import queue
from threading import Thread
from typing import Any, Callable, ContextManager, Mapping, Optional, Sequence

log = logging.getLogger(__name__)
STOP = -1
STOP_TASK = (
    STOP,
    STOP,
    STOP,
)


class Queue(queue.Queue):
    def clear(self):
        with self.mutex:
            unfinished_tasks = self.unfinished_tasks - len(self.queue)
            if unfinished_tasks <= 0:
                if unfinished_tasks < 0:
                    raise ValueError("task_done() called too many times")
                self.all_tasks_done.notify_all()
            self.unfinished_tasks = unfinished_tasks
            self.queue.clear()
            self.not_full.notify_all()


class AsyncTaskManager(ContextManager):
    """
    Multithreaded task manager.
    Not that the manager itself is not thread safe - meaning it can be used only in a single thread.
    """

    def __init__(self, workers_amount: int = 5, iter_timeout: Optional[float] = None):
        super().__init__()
        self.iter_timeout = iter_timeout
        self._task_queue: Queue = Queue()
        self._result_queue: Queue = Queue()
        self._total_task_count = 0
        self._workers_count = 0
        self.start_workers(workers_amount)

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_work_done and self.is_empty:
            raise StopIteration()
        return self.get_result(timeout=self.iter_timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        self.kill()

    @property
    def is_work_done(self) -> bool:
        """
        :return: True iff all tasks processing is done.
        """
        return self._task_queue.unfinished_tasks == 0

    @property
    def is_empty(self) -> bool:
        """
        :return: True iff all results fetching is done.
        """
        return self._result_queue.unfinished_tasks == 0

    @property
    def is_closed(self) -> bool:
        """
        :return: True iff all worker threads are closed.
        """
        return self._workers_count == 0

    @property
    def total_task_count(self) -> int:
        return self._total_task_count

    def start_workers(self, workers_amount: int):
        log.debug(f"Starting {workers_amount} workers.")
        for i in range(workers_amount):
            thread = Thread(target=self._work, daemon=True)
            thread.start()
            self._workers_count += 1

    def add_task(self, func: Callable, args: Sequence = None, kwargs: Mapping = None):
        if self.is_closed:
            return
        args = tuple(args) if args else ()
        kwargs = kwargs or {}
        self._total_task_count += 1
        self._task_queue.put((func, args, kwargs))

    def _work(self):
        log.debug("Worker starting.")
        while True:
            func, args, kwargs = self._task_queue.get()
            if func == STOP:
                self._task_queue.task_done()
                break
            result = func(*args, **kwargs)
            self._result_queue.put(result)
            self._task_queue.task_done()
        log.debug("Worker done.")

    def get_result(self, timeout: Optional[float] = 3) -> Any:
        result = self._result_queue.get(block=True, timeout=timeout)
        self._result_queue.task_done()
        return result

    def stop_workers(self, amount: int = None):
        if amount is None:
            amount = self._workers_count
        log.debug(f"Stopping {amount} workers.")
        for i in range(amount):
            self._task_queue.put(STOP_TASK)
            self._workers_count -= 1

    def kill(self):
        log.debug(f"Removing all items from task queue (current size is {self._task_queue.qsize()})...")
        self._task_queue.clear()
        self.join()

    def join(self):
        log.debug("Joining all workers...")
        self._task_queue.join()  # Wait until all work is done
        self.stop_workers()
        self._task_queue.join()  # Wait until all workers are dead
