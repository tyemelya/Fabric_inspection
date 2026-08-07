import pytest_asyncio, pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from unittest.mock import patch
from app.api.schemas import InspectionResponse

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
        
@pytest.mark.asyncio
async def test_get_image(client):
    response = await client.get("/images/1")
    
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

@pytest.mark.asyncio
async def test_get_invalid_image(client):
    response = await client.get("/images/-1")

    assert response.status_code == 404

@pytest.mark.asyncio
async def test_health(client):
        response = await client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok"
        }

@pytest.mark.asyncio
@patch("app.api.routes.inspection_service.inspect")
async def test_inspect(mock_inspect, client):
    mock_inspect.return_value = {
        "report": "Defect-free fabric.",
        "vision_result": type(
            "Vision",
            (),
            {
                "prediction": "defect free",
                "confidence": 0.99,
            },
        )(),
        "evidence": None,
        "retrieved_cases": [],
    }    
    
    with open("tests/data/sample.jpg", "rb") as f:
        response = await client.post(
            "/inspect",
            files={"uploaded_image": ("test.jpg", f, "image/jpeg")},
            data={"question": "What defect?"},
        )

    assert response.status_code == 200
    assert response.json()["prediction"] == "defect free"