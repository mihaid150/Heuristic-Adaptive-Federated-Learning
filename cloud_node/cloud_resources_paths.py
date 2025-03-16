from enum import Enum


class CloudResourcesPaths(str, Enum):
    MODELS_FOLDER_PATH = "/app/models/"
    RESULTS_FOLDER_PATH = "/app/results/"
    CLOUD_MODEL_FILE_NAME = "cloud_model.keras"
    FOG_MODEL_FILE_NAME = "fog_model_id_{child_id}.keras"
    DB_RESULTS_FILE = "db_results.db"

