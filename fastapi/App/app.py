from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model.predict import MLFLOW_MODEL_VERSION , predict_output
from schema.user_input import UserInput
from schema.output_schema import OutputSchema

app = FastAPI(
    title="Kidney Disease Classifier API",
    version="1.0.0"
)




@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version" : MLFLOW_MODEL_VERSION
    }


@app.post("/predict",response_model=OutputSchema)
def predict(file:UploadFile):

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
        confidence , predicted_class = predict_output(input=file.file)

        return JSONResponse(status_code=200,content= {
            "prediction": predicted_class,
            "confidence": confidence
        })

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )