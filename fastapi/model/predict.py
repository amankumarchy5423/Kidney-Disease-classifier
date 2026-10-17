
from pathlib import Path

from tensorflow import keras
from PIL import Image
import numpy as np


MLFLOW_MODEL_VERSION = "1.0.0"


MODEL_PATH = Path(__file__).resolve().parent / (
    "production_kidney_classifier_model.keras"
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}"
    )

model = keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Cyst",
    "Normal",
    "Stone",
    "Tumor",
]


def predict_output(input):
    try:
        with Image.open(input) as image:
            image = image.convert("RGB")
            image = image.resize((224, 224))
            image = np.array(image, dtype=np.float32)

        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image, verbose=0)

        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_index])
        predicted_class = CLASS_NAMES[predicted_index]

        return confidence, predicted_class

    except Exception as e:
        raise e
