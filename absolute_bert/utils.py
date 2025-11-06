
from collections.abc import Callable
from functools import partial
from typing import Generic, TypeVar
import logging
import os

logger = logging.getLogger(__name__)

N = TypeVar("N", bound=str)
T = TypeVar("T")


class Registry(Generic[N, T]):
    def __init__(self) -> None:
        self._registry: dict[N, T] = {}

    def register(self, key: N) -> Callable[[T], None]:
        return partial(self._register, key)

    def _register(self, key: N, item: T) -> None:
        self._registry[key] = item

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