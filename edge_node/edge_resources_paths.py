from enum import Enum


class EdgeResourcesPaths(str, Enum):
    MODELS_FOLDER_PATH = "/app/models/"
    EDGE_MODEL_FILE_NAME = "edge_model.keras"
    RETRAINED_EDGE_MODEL_FILE_NAME = "retrained_edge_model.keras"

    DATA_FOLDER_PATH = "/app/data/"
    FILTERED_DATA_FOLDER_PATH = "/app/data/filtered_data/"
    INPUT_DATA_PATH = DATA_FOLDER_PATH + "input_data.csv"
    FILTERED_DATA_PATH = FILTERED_DATA_FOLDER_PATH + "filtered_data.csv"

    CURRENT_DAY_DATA_PATH = FILTERED_DATA_FOLDER_PATH + "current_day_data.csv"
    NEXT_DAY_DATA_PATH = FILTERED_DATA_FOLDER_PATH + "next_day_data.csv"
