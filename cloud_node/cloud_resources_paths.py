from enum import Enum


class CloudResourcesPaths(str, Enum):
    MODELS_FOLDER_PATH = "/app/models/"
    CLOUD_MODEL_FILE_NAME = "cloud_model.keras"
    FOG_MODEL_FILE_NAME = "fog_model_id_{child_id}.keras"
    MODEL_PERFORMANCE_RESULTS_FILE_NAME = "model_performance.json"

