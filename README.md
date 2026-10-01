# Real-Time ML Inference REST API & Capstone

A production-style machine learning inference API built with FastAPI. The project exposes a `/predict` endpoint that accepts JSON input and returns a prediction with class probabilities.

## Architecture

Client → FastAPI `/predict` → Trained Random Forest Model → Prediction + Probabilities → JSON Response

## Project Structure

```text
task6_ml_rest_api/
├── app/
│   └── main.py
├── model/
│   └── model.joblib
├── tests/
│   └── test_api.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn app.main:app --reload
```

### 3. Open Swagger UI

Visit:

`http://127.0.0.1:8000/docs`

### 4. Test `/predict`

Use this JSON:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The response contains the predicted class, label, and prediction probabilities.

## Run Tests

```bash
pytest -q
```

## Docker

Build:

```bash
docker build -t task6-ml-api .
```

Run:

```bash
docker run -p 8000:8000 task6-ml-api
```

Then open:

`http://127.0.0.1:8000/docs`

## Implementation Notes

- FastAPI provides the REST API.
- Pydantic validates incoming JSON payloads.
- A Random Forest classifier is used as the trained champion model.
- Joblib loads the model at application startup.
- `/predict` returns prediction probabilities.
- `/health` provides a basic health check.
- Pytest validates the prediction schema, response status, and invalid input handling.
- Docker packages the service with pinned Python dependencies.

## Submission

Push this entire folder to a public GitHub repository and submit the repository URL as the single public link.
