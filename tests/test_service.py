from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from fracturelens.service.app import app


def test_web_app_is_available_without_loading_models() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "FractureLens" in response.text
    assert 'id="image-input"' in response.text
    assert '"fracture_signal_with_local_evidence"' in response.text
    assert '"no_fracture_signal"' in response.text
    assert 'window.location.protocol === "file:"' in response.text
    assert 'id="download-button"' in response.text
    assert "Model uyuşmazlık uyarısı" in response.text


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


def test_predict_accepts_swagger_style_file_upload(monkeypatch) -> None:
    class FakePipeline:
        def predict(self, image: Image.Image, image_id: str | None = None) -> dict:
            return {"width": image.width, "height": image.height, "image_id": image_id}

    monkeypatch.setattr("fracturelens.service.app.get_pipeline", lambda: FakePipeline())
    image_bytes = BytesIO()
    Image.new("RGB", (12, 8)).save(image_bytes, format="PNG")

    response = TestClient(app).post(
        "/v1/predict",
        files={"file": ("xray.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {"width": 12, "height": 8, "image_id": "xray.png"}
