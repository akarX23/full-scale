from typing import Dict, FrozenSet

SUPPORTED_MODELS: FrozenSet[str] = frozenset({"lenet", "alexnet", "resnet18", "vgg16"})

OPTIMAL_BATCH_SIZE: Dict[str, int] = {"NPU": 8, "GPU": 16, "CPU": 6}


DEFAULT_MODEL: str      = "lenet"
DEFAULT_MODEL_TYPE: str = "dense"
MAX_TOP_K: int          = 50
DEFAULT_MODEL_DIR: str  = "./models"
DEFAULT_DATA_DIR: str   = "./data"
DEFAULT_DEVICE: str     = "CPU"
DEFAULT_HOST: str       = "0.0.0.0"
DEFAULT_PORT: int       = 8000
AUTO_BATCH_TIMEOUT: float = 0.5  # seconds
ADD_OPTIMAL_REQS: int = 2        # extra infer requests on top of OPTIMAL_NUMBER_OF_INFER_REQUESTS
