
from collections.abc import Callable
from functools import partial
from typing import Generic, TypeVar, get_args
import logging
import os

logger = logging.getLogger(__name__)

N = TypeVar("N", bound=str)
T = TypeVar("T")


class Registry(Generic[N, T]):
    def __init__(self) -> None:
        self._registry: dict[N, T] = {}

    def __repr__(self) -> str:
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            name_type = get_args(self.__orig_class__)[0]
            return f"<registry[{str(name_type)}]: {self._registry.keys()}>"
        
        orig_bases = getattr(self, "__orig_bases__", None)
        if orig_bases is not None:
            for base in orig_bases:
                if not isinstance(base, Registry):
                    continue

                name_type = get_args(base)[0]
                return f"<registry[{str(name_type)}]: {self._registry.keys()}>"
            
        return f"<registry[Unknown]: {self._registry.keys()}>"

    def register(self, key: N) -> Callable[[T], T]:
        return partial(self._register, key)

    def _register(self, key: N, item: T) -> T:
        logger.debug(f"{str(self)} registering, `{key=}, {item=}`")
        self._registry[key] = item
        return item

    def get(self, key: N) -> T | None:
        return self._registry.get(key)

    def __getitem__(self, key: N) -> T:
        return self._registry.__getitem__(key)
    



def init_logging(level: str | None = None):

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    level_int = getattr(logging, level.upper())

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level_int,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logging.info(f"set log level to {level_int}({level})")

    debugging_module = os.getenv("DEBUGGING_MODULE", None)
    if debugging_module is not None:
        logging.getLogger(debugging_module).setLevel(logging.DEBUG)