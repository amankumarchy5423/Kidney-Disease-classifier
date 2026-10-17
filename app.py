import base64
import io
import os

import requests
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "Frontend")

# Where your FastAPI backend is running
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error="The image is larger than 8 MB."), 413


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Choose a CT image to analyse.")
    if not allowed_file(file.filename):
        return render_template("index.html", error="Unsupported file. Upload a PNG or JPG image.")

    image_bytes = file.read()
    mimetype = file.mimetype
    # Shown back on the page, so nothing is saved to disk
    image_data = f"data:{mimetype};base64,{base64.b64encode(image_bytes).decode()}"

    try:
        response = requests.post(
            f"{FASTAPI_URL}/predict",
            files={"file": (file.filename, io.BytesIO(image_bytes), mimetype)},
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        return render_template("index.html", image_data=image_data,
                               error="Cannot reach the model server. Start the FastAPI backend and try again.")
    except requests.exceptions.Timeout:
        return render_template("index.html", image_data=image_data,
                               error="The model server took too long to respond.")

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "Prediction failed.")
        except ValueError:
            detail = "Prediction failed."
        return render_template("index.html", image_data=image_data, error=str(detail))

    data = response.json()
    confidence = float(data["confidence"])
    if confidence <= 1:          # backend sent a fraction (0.97) instead of a percentage (97)
        confidence *= 100

    result = {"label": data["prediction"], "confidence": round(confidence, 2)}
    return render_template("index.html", image_data=image_data, result=result)


if __name__ == "__main__":
    app.run(port=5000, debug=True)