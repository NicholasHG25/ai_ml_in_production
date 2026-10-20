from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_positive():
    r = client.post("/predict", json={"text": "Hotelnya bagus dan pelayanannya mantap"})
    assert r.status_code == 200
    assert r.json()["sentiment"] == "positif"


def test_negative():
    r = client.post("/predict", json={"text": "Pelayanan buruk dan kecewa"})
    assert r.json()["sentiment"] == "negatif"


def test_validation_error():
    r = client.post("/predict", json={})
    assert r.status_code == 422
