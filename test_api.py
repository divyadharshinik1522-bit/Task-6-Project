from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_schema():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_class" in data
    assert "predicted_label" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 3

def test_invalid_payload():
    response = client.post("/predict", json={"sepal_length": -1})
    assert response.status_code == 422
