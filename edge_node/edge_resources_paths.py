from enum import Enum


class EdgeResourcesPaths(str, Enum):
    MODELS_FOLDER_PATH = "/app/models/"
    EDGE_MODEL_FILE_NAME = "edge_model.keras"
    RETRAINED_EDGE_MODEL_FILE_NAME = "retrained_edge_model.keras"

    DATA_FOLDER_PATH = "/app/data/"
    FILTERED_DATA_FOLDER_PATH = "/app/data/filtered_data/"
    INPUT_DATA_PATH = DATA_FOLDER_PATH + "input_data.csv"
    FILTERED_DATA_PATH = FILTERED_DATA_FOLDER_PATH + "filtered_data.csv"

    RETRAINING_DAYS_DATA_PATH = FILTERED_DATA_FOLDER_PATH + "retraining_days_data.csv"
    EVALUATION_DAYS_DATA_PATH = FILTERED_DATA_FOLDER_PATH + "evaluation_days_data.csv"

    FLAGS_FOLDER_PATH = "/app/flags/"
    COMPLETED_PREVIOUS_ROUND_JSON_FILE = FLAGS_FOLDER_PATH + "completed_previous_round.json"
