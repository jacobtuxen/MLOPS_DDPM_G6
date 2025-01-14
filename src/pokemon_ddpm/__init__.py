import os
from pathlib import Path

__PATH_TO_ROOT__ = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parent.parent
__PATH_TO_DATA__ = Path(os.path.join(__PATH_TO_ROOT__, "data"))
