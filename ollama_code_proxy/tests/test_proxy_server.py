from fastapi.testclient import TestClient
from fastapi import FastAPI, Response as FastAPIResponse
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx # For httpx.HTTPStatusError

from ollama_code_proxy.proxy_server import routes as proxy_routes_module
# Import models if needed for constructing request data, though for these tests, dicts are mostly used.
# from ollama_code_proxy.proxy_server import models as proxy_models_module

# Create a minimal app for testing
app = FastAPI()
# The router in proxy_routes_module should be defined WITHOUT a prefix.
# The prefix '/api/v1/ollama' is applied here when including the router.
app.include_router(proxy_routes_module.router, prefix="/api/v1/ollama")

client = TestClient(app)

# --- Test for Root Endpoint ---
def test_read_root_on_test_app():
    # This test app only has routes starting with /api/v1/ollama
    # So, a request to "/" should result in a 404 Not Found.
    response = client.get("/")
    assert response.status_code == 404


# --- Mocks for Ollama Responses ---
def mock_ollama_generate_response_data():
    return {
        "model": "test-model", "created_at": "2023-01-01T00:00:00Z",
        "response": "Test response from Ollama", "done": True, "context": [1, 2, 3],
        "total_duration": 1000, "load_duration": 100, "prompt_eval_count": 10,
        "prompt_eval_duration": 200, "eval_count": 5, "eval_duration": 300,
    }

def mock_ollama_tags_response_data():
    return {
        "models": [{
            "name": "llama2:latest", "model": "llama2:latest", "modified_at": "2023-01-01T00:00:00Z",
            "size": 1234567890, "digest": "sha256:abcdef",
            "details": {"parent_model": "", "format": "gguf", "family": "llama",
                        "families": ["llama"], "parameter_size": "7B", "quantization_level": "Q4_0"}
        }]
    }

def mock_ollama_show_response_data():
    return {
        "modelfile": "FROM llama2...", "parameters": "num_ctx 4096", "template": "{{ .Prompt }}",
        "details": {"parent_model": "", "format": "gguf", "family": "llama",
                    "families": ["llama"], "parameter_size": "7B", "quantization_level": "Q4_0"},
        "model_info": {"general.architecture": "llama", "llama.block_count": 32}
    }

def mock_ollama_chat_response_data(streaming=False):
    if streaming:
        return {"model": "chat-model", "created_at": "2023-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "Stream part..."}, "done": False}
    return {"model": "chat-model", "created_at": "2023-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Full chat response"}, "done": True,
            "total_duration": 1000, "prompt_eval_count": 10, "eval_count": 5}

def mock_ollama_pull_status_response_data(streaming=False):
    if streaming:
        return {"status": "downloading layer sha256:12345", "digest": "sha256:12345", "total": 1000, "completed": 500}
    return {"status": "success"} # Non-streaming success for pull


# --- Tests for /api/v1/ollama/generate ---
@patch("ollama_code_proxy.proxy_server.routes._ollama_proxy_request", new_callable=AsyncMock)
def test_proxy_generate_success(mock_proxy_helper):
    mock_proxy_helper.return_value = mock_ollama_generate_response_data()
    # Default PROMPT_PREFIX is "Based on your knowledge, please respond to the following: \n"
    expected_prefixed_prompt = "Based on your knowledge, please respond to the following: \nTest prompt"

    request_data = {"model": "test-model", "prompt": "Test prompt", "stream": False}
    response = client.post("/api/v1/ollama/generate", json=request_data)

    assert response.status_code == 200
    assert response.json()["response"] == "Test response from Ollama"

    mock_proxy_helper.assert_called_once()
    args, kwargs = mock_proxy_helper.call_args
    assert args[0] == "POST"
    assert args[1] == "/api/generate"
    # args[2] is the settings object, we can't easily assert its exact instance from here without more setup
    # So we check the json_data that was passed, which is what we care about for this test
    assert "json_data" in kwargs
    assert kwargs["json_data"]["prompt"] == expected_prefixed_prompt
    assert kwargs["json_data"]["model"] == "test-model"


@patch("ollama_code_proxy.proxy_server.routes._ollama_proxy_request", new_callable=AsyncMock)
def test_proxy_generate_ollama_error(mock_proxy_helper):
    # Simulate the helper re-raising an HTTPException that originated from an Ollama error
    mock_proxy_helper.side_effect = HTTPException(status_code=500, detail={"error": "ollama internal error"})

    request_data = {"model": "test-model", "prompt": "Test prompt", "stream": False}
    response = client.post("/api/v1/ollama/generate", json=request_data)

    assert response.status_code == 500
    assert response.json()["detail"] == {"error": "ollama internal error"}

# TODO: Add test for /generate streaming


# --- Tests for /api/v1/ollama/tags ---
@patch("ollama_code_proxy.proxy_server.routes._ollama_proxy_request", new_callable=AsyncMock)
def test_proxy_tags_success(mock_proxy_helper):
    mock_proxy_helper.return_value = mock_ollama_tags_response_data()
    response = client.get("/api/v1/ollama/tags")
    assert response.status_code == 200
    assert len(response.json()["models"]) == 1
    assert response.json()["models"][0]["name"] == "llama2:latest"

    mock_proxy_helper.assert_called_once()
    args, kwargs = mock_proxy_helper.call_args
    assert args[0] == "GET"
    assert args[1] == "/api/tags"
    # args[2] is settings

# --- Tests for /api/v1/ollama/show ---
@patch("ollama_code_proxy.proxy_server.routes._ollama_proxy_request", new_callable=AsyncMock)
def test_proxy_show_success(mock_proxy_helper):
    mock_proxy_helper.return_value = mock_ollama_show_response_data()
    request_data = {"name": "llama2:latest"} # This matches ShowRequest model
    response = client.post("/api/v1/ollama/show", json=request_data)
    assert response.status_code == 200
    assert response.json()["modelfile"] == "FROM llama2..."

    mock_proxy_helper.assert_called_once()
    args, kwargs = mock_proxy_helper.call_args
    assert args[0] == "POST"
    assert args[1] == "/api/show"
    assert kwargs["json_data"] == request_data

# --- Tests for /api/v1/ollama/chat (non-streaming) ---
@patch("ollama_code_proxy.proxy_server.routes._ollama_proxy_request", new_callable=AsyncMock)
def test_proxy_chat_non_streaming_success(mock_proxy_helper):
    mock_proxy_helper.return_value = mock_ollama_chat_response_data(streaming=False)
    request_data = {
        "model": "chat-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False # Explicitly non-streaming
    }
    response = client.post("/api/v1/ollama/chat", json=request_data)
    assert response.status_code == 200
    assert response.json()["message"]["content"] == "Full chat response"

    mock_proxy_helper.assert_called_once()
    args, kwargs = mock_proxy_helper.call_args
    assert args[0] == "POST"
    assert args[1] == "/api/chat"
    assert kwargs["json_data"] == request_data

# TODO: Add test for /chat streaming

# --- Tests for /api/v1/ollama/pull (non-streaming) ---
@patch("ollama_code_proxy.proxy_server.routes._ollama_proxy_request", new_callable=AsyncMock)
def test_proxy_pull_non_streaming_success(mock_proxy_helper):
    mock_proxy_helper.return_value = mock_ollama_pull_status_response_data(streaming=False)
    request_data = {"name": "llama2:latest", "stream": False} # Explicitly non-streaming
    response = client.post("/api/v1/ollama/pull", json=request_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    mock_proxy_helper.assert_called_once()
    args, kwargs = mock_proxy_helper.call_args
    assert args[0] == "POST"
    assert args[1] == "/api/pull"
    assert kwargs["json_data"] == request_data

# TODO: Add test for /pull streaming

# --- Tests for /api/v1/ollama/delete ---
@patch("ollama_code_proxy.proxy_server.routes._ollama_proxy_request", new_callable=AsyncMock)
def test_proxy_delete_success(mock_proxy_helper):
    # _ollama_proxy_request for a successful delete with no content returns a FastAPIResponse(status_code=200)
    mock_proxy_helper.return_value = FastAPIResponse(status_code=200)
    request_data = {"name": "model-to-delete"}
    response = client.delete("/api/v1/ollama/delete", json=request_data)
    assert response.status_code == 200
    assert response.content == b"" # Expecting empty body for 200 from helper

    mock_proxy_helper.assert_called_once()
    args, kwargs = mock_proxy_helper.call_args
    assert args[0] == "DELETE"
    assert args[1] == "/api/delete"
    assert kwargs["json_data"] == request_data
    assert kwargs["expected_empty_response_status"] == 200


# --- Tests for /api/v1/ollama/ps ---
@patch("ollama_code_proxy.proxy_server.routes._ollama_proxy_request", new_callable=AsyncMock)
def test_proxy_ps_success(mock_proxy_helper):
    # Using a more complete mock based on ProcessModelInfo structure
    mock_ps_model_details = {"parent_model": "", "format": "gguf", "family": "llama",
                             "families": ["llama"], "parameter_size": "7B", "quantization_level": "Q4_0"}
    mock_response_data = {
        "models": [{
            "name": "llama2:loaded", "model": "llama2:loaded", "modified_at": "2023-12-01T00:00:00Z",
            "size": 7000000, "digest":"abc", "details": mock_ps_model_details,
            "expires_at": "2024-01-01T00:00:00Z", "size_vram": 123456
        }]
    }
    mock_proxy_helper.return_value = mock_response_data
    response = client.get("/api/v1/ollama/ps")
    assert response.status_code == 200
    assert len(response.json()["models"]) == 1
    assert response.json()["models"][0]["name"] == "llama2:loaded"

    mock_proxy_helper.assert_called_once()
    args, kwargs = mock_proxy_helper.call_args
    assert args[0] == "GET"
    assert args[1] == "/api/ps"
