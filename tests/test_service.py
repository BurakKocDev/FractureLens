from fastapi.testclient import TestClient

from fracturelens.service.app import app


def test_health_does_not_load_models() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "not for diagnosis" in response.json()["disclaimer"]


def test_predict_rejects_non_image_content_type() -> None:
    response = TestClient(app).post(
        "/v1/predict", content=b"not an image", headers={"content-type": "text/plain"}
    )

    assert response.status_code == 415
