from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
DATA_FILE_PATH = Path("C:/Users/Public/Documents/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")

# Data Ingestion related variables
ARTIFACT_DIR = Path("artifact")
DATA_INGESTION_DIR = Path("data_ingestion")
TRAIN_FILE = "train"
TEST_FILE = "test"
VAL_FILE = "validation"

#MODEL building related directory
MODEL_BUILDING = 'model_building'

# model training 
MODEL_TRAINING = 'model_training'
MODEL_STUCTURE_PNG = 'model_structure.png'
MODEL_FILE = "model.keras"