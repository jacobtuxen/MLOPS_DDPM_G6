import os
from pathlib import Path

_PATH_TO_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parent.parent
_PATH_TO_DATA = Path(os.path.join(_PATH_TO_ROOT, "data"))
_PATH_TO_MODELS = Path(os.path.join(_PATH_TO_ROOT, "models"))
_PATH_TO_CONFIG = Path(os.path.join(_PATH_TO_ROOT, "configs"))
