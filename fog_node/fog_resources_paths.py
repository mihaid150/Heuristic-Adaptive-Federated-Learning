from enum import Enum


class FogResourcesPaths(str, Enum):
    MODELS_FOLDER_PATH = "/app/models/"
    FOG_MODEL_FILE_NAME = "fog_model.keras"
    EDGE_MODEL_FILE_NAME = "edge_model.keras"
    LAMBDA_PREV_FILE_NAME = "lambda_prev.json"
