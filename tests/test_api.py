from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_predict_endpoint_ok():
    payload = {
        "length1": 23.2,
        "length2": 25.4,
        "length3": 30.0,
        "height": 11.52,
        "width": 4.02,
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "predicted_weight" in data
    assert isinstance(data["predicted_weight"], float)
    assert data["predicted_weight"] > 0
