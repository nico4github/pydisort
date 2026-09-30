from importlib.metadata import PackageNotFoundError, version

import torch  # noqa: F401

from .pydisort import *
from .timing import TimingCollector as TimingCollector
from .timing import TimingRecord as TimingRecord
from .timing import timed as timed

try:
    __version__ = version("pydisort")
except PackageNotFoundError:
    __version__ = "0.0.0"
