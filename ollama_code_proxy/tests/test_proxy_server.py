from fastapi.testclient import TestClient
from main import app
from proxy_server.models import OllamaRequest, OllamaResponse
import pytest
import httpx # Import httpx

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Ollama Code Proxy is running!"}

def test_proxy_to_ollama_success(mocker):
    # Mock the httpx.AsyncClient response
    mock_response_data = {
        "model": "test-model",
        "created_at": "2023-01-01T00:00:00Z",
        "response": "Test response from Ollama",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 1000,
        "load_duration": 100,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200,
        "eval_count": 5,
        "eval_duration": 300,
    }
    # aparición de httpx.AsyncClient.post
    mock_async_client_post = mocker.patch("httpx.AsyncClient.post")
    # Crear un mock para el objeto de respuesta
    mock_http_response = mocker.Mock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = mock_response_data
    # Configurar raise_for_status para no hacer nada (simulando una respuesta exitosa)
    mock_http_response.raise_for_status = mocker.Mock()
    mock_async_client_post.return_value = mock_http_response


    request_data = OllamaRequest(model="test-model", prompt="Test prompt")
    response = client.post("/api/v1/ollama/generate", json=request_data.model_dump())

    assert response.status_code == 200
    assert response.json() == mock_response_data
    mock_async_client_post.assert_called_once()

def test_proxy_to_ollama_http_error(mocker):
    # Mock an HTTP error from httpx by having post raise it directly
    # We need to create a mock request object for the HTTPStatusError
    mock_request = httpx.Request(method="POST", url="http://localhost:11434/api/generate")
    # We also need a mock response object for the HTTPStatusError
    mock_response = httpx.Response(status_code=500, request=mock_request, content=b"Ollama Error")

    mocker.patch(
        "httpx.AsyncClient.post",
        side_effect=httpx.HTTPStatusError(
            "Ollama Server Error", request=mock_request, response=mock_response
        ),
    )

    request_data = OllamaRequest(model="test-model", prompt="Test prompt")
    response = client.post("/api/v1/ollama/generate", json=request_data.model_dump())

    assert response.status_code == 500
    assert "Ollama Server Error" in response.json()["detail"]


def test_proxy_to_ollama_request_error(mocker):
    # Mock a request error (e.g., connection issue)
    # We need to create a mock request object for the RequestError
    mock_request = httpx.Request(method="POST", url="http://localhost:11434/api/generate")
    mocker.patch(
        "httpx.AsyncClient.post",
        side_effect=httpx.RequestError("Connection failed", request=mock_request),
    )

    request_data = OllamaRequest(model="test-model", prompt="Test prompt")
    response = client.post("/api/v1/ollama/generate", json=request_data.model_dump())

    assert response.status_code == 500
    assert "Error connecting to Ollama API: Connection failed" in response.json()["detail"]
