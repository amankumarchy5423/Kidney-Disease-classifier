FROM python:3.12-bookworm

WORKDIR /app

# Install dependencies
COPY fastapi-requirements.txt .
RUN pip install --no-cache-dir -r fastapi-requirements.txt

COPY . .

# Run the FastAPI app (App/app.py -> App.app)
CMD ["uvicorn", "App.app:app", "--host", "0.0.0.0", "--port", "8080"]