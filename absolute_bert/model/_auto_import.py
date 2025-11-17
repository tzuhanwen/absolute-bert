# This module will import modules of the same level of this module,
# __init__.py file and explicitly import submodules in the file is needed.

import pkgutil
import importlib
from pathlib import Path

from .registry import lm_registry, lm_config_registry

import logging

logger = logging.getLogger(__name__)
logger.debug("start to auto import")

_pkg = Path(__file__).parent
logger.debug(f"{_pkg=}")


for m in pkgutil.iter_modules([str(_pkg)]):
    logger.debug(f"found module {m}")

    name = m.name
    if name.startswith("_") or name in {"registry.py"}:
        continue

    logger.debug(f"importing `{name}`")

    importlib.import_module(f"{__package__}.{name}")

logger.debug(
    "end of auto import, "
    f"{lm_config_registry._registry.keys()=}, "
    f"{lm_registry._registry.keys()=}"
)
