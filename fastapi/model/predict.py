from tensorflow import keras
from PIL import Image
import numpy as np
from fastapi import File



MLFLOW_MODEL_VERSION = '1.0.0'
MODEL_PATH = r"C:\Users\CS_\OneDrive\Desktop\project\Kidney-Disease-classifier\artifact\model_training\model.keras"

model = keras.models.load_model(MODEL_PATH)


CLASS_NAMES = [
    "Cyst",
    "Normal",
    "Stone",
    "Tumor"
]

def predict_output(input ):
    try :
         
        image = Image.open(input).convert("RGB")

        image = image.resize((224, 224))

        image = np.array(image, dtype=np.float32)

        image = image / 255.0

        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)

        predicted_index = int(
            np.argmax(prediction[0])
        )

        confidence = float(
            prediction[0][predicted_index]
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        return confidence , predicted_class
            
    except Exception as e :
        raise e 