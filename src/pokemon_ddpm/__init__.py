import os
from pathlib import Path

_PATH_TO_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).resolve().parent.parent.parent))
_PATH_TO_DATA = Path(os.path.join(_PATH_TO_ROOT, "data"))
_PATH_TO_MODELS = Path(os.path.join(_PATH_TO_ROOT, "models"))
