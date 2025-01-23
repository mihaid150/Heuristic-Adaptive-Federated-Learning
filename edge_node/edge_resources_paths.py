from enum import Enum


class EdgeResourcesPaths(str, Enum):
    MODELS_FOLDER_PATH = "/app/models/"
    EDGE_MODEL_FILE_NAME = "edge_model.keras"
    RETRAINED_EDGE_MODEL_FILE_NAME = "retrained_edge_model.keras"

    DATA_FOLDER_PATH = "/app/data"
    INPUT_DATA_PATH = "lorem ipsum"
    FILTERED_DATA_PATH = "lorem ipsum"

    CURRENT_DAY_DATA_PATH = "lorem ipsum"
    NEXT_DAY_DATA_PATH = "lorem ipsum"
