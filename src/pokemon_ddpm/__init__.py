import os
from pathlib import Path

_PATH_TO_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
_PATH_TO_DATA = Path(os.path.join(_PATH_TO_ROOT, "data"))
_PATH_TO_MODELS = Path(os.path.join(_PATH_TO_ROOT, "models"))
_PATH_TO_CONFIG = Path(os.path.join(_PATH_TO_ROOT, "configs"))
_PATH_TO_OUTPUT = Path(os.path.join(_PATH_TO_ROOT, "outputs"))
_PATH_TO_SWEEP = Path(os.path.join(_PATH_TO_CONFIG, "sweep.yaml"))
