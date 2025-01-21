from enum import Enum


class CloudResourcesPaths(str, Enum):
    CLOUD_MODEL_FOLDER_PATH = "/app/models/"
    CLOUD_MODEL_FILE_NAME = "cloud_model.keras"
