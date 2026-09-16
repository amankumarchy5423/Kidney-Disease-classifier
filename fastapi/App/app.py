from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow import keras
from PIL import Image
import numpy as np


app = FastAPI(
    title="Kidney Disease Classifier API",
    version="1.0.0"
)


MODEL_PATH = "artifact/model_training/model.keras"

model = keras.models.load_model(MODEL_PATH)


CLASS_NAMES = [
    "Cyst",
    "Normal",
    "Stone",
    "Tumor"
]


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        if file.content_type not in [
            "image/jpeg",
            "image/png",
            "image/jpg"
        ]:
            raise HTTPException(
                status_code=400,
                detail="Only JPG and PNG images are supported."
            )

        image = Image.open(file.file).convert("RGB")

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

        return {
            "prediction": predicted_class,
            "confidence": confidence
        }

    

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )