import importlib.metadata
from pathlib import Path

import toml

from .core import *
from .process_video import *
from .alphapose import *
from .overlay import *
from .labelbox import *
from .utils import *

PROJECT_NAME = 'wf-pose-labelbox' # Keep this synced with project name in pyproject.toml

def get_version() -> str:
    try:
        version = importlib.metadata.version(PROJECT_NAME)
    except:
        path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        pyproject = toml.load(str(path))
        version: str = pyproject["tool"]["poetry"]["version"]
    return version


__version__ = get_version()
