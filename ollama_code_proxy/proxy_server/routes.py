from fastapi import APIRouter, HTTPException, Depends, Body, Response
from fastapi.responses import StreamingResponse
import httpx
import os
import json
from typing import Any, Dict, Optional, Union, AsyncGenerator

from .models import (
    OllamaRequest, OllamaResponse, TagsResponse, ShowRequest, ShowResponse,
    ChatMessage, ChatRequest, ChatResponse, PullRequest, PullStatus,
    DeleteRequest, CopyRequest, EmbedRequest, EmbedResponse, PsResponse,
    VersionResponse, CreateRequest, StatusOkResponse
)

router = APIRouter() # Prefix is handled in main.py

# --- Configuration ---
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL_DEFAULT)

PROMPT_PREFIX_DEFAULT = "Based on your knowledge, please respond to the following: \n"
PROMPT_PREFIX = os.getenv("PROMPT_PREFIX", PROMPT_PREFIX_DEFAULT)

class Settings:
    ollama_base_url: str = OLLAMA_BASE_URL
    prompt_prefix: str = PROMPT_PREFIX

def get_settings():
    return Settings()

# --- Helper for HTTP client (primarily for non-streaming GET/POST/DELETE) ---
async def _ollama_proxy_request(
    method: str,
    ollama_endpoint: str,
    settings: Settings,
    json_data: Optional[Dict[str, Any]] = None,
    params_data: Optional[Dict[str, Any]] = None,
    expected_empty_response_status: Optional[int] = None
) -> Any:
    full_ollama_url = f"{settings.ollama_base_url}{ollama_endpoint}"
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            if method.upper() == "GET":
                response = await client.get(full_ollama_url, params=params_data)
            elif method.upper() == "POST":
                response = await client.post(full_ollama_url, json=json_data, params=params_data)
            elif method.upper() == "DELETE":
                response = await client.request("DELETE", full_ollama_url, json=json_data, params=params_data)
            else:
                raise HTTPException(status_code=501, detail=f"Unsupported proxy method: {method}")

            response.raise_for_status()

            if expected_empty_response_status and response.status_code == expected_empty_response_status:
                return Response(status_code=expected_empty_response_status)

            if not response.content and (200 <= response.status_code <= 299):
                 return Response(status_code=response.status_code)

            return response.json()
    except httpx.HTTPStatusError as e:
        error_detail = str(e)
        try:
            if e.response and e.response.text: error_detail = e.response.json()
        except Exception:
            if e.response and e.response.text: error_detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error connecting to Ollama API at {full_ollama_url}: {str(e)}")
    except Exception as e:
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Helper for Streaming Responses ---
async def _stream_ollama_response(
    ollama_url: str,
    payload: Dict[str, Any],
    http_method: str = "POST"
) -> AsyncGenerator[str, None]:
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(http_method, ollama_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield f"{line}\n"
    except httpx.HTTPStatusError as e:
        error_content = {"error": "Ollama API error during stream setup", "status_code": e.response.status_code if e.response else 500}
        try:
            error_content["detail"] = e.response.json() if e.response else str(e)
        except json.JSONDecodeError:
            error_content["detail"] = e.response.text if e.response and e.response.text else str(e)
        except Exception as detail_ex:
             error_content["detail"] = str(detail_ex)
        yield f"{json.dumps(error_content)}\n"
    except httpx.RequestError as e:
        error_content = {"error": "Connection to Ollama API failed during stream", "detail": str(e)}
        yield f"{json.dumps(error_content)}\n"
    except Exception as e:
        error_content = {"error": "Streaming failed unexpectedly", "detail": str(e)}
        yield f"{json.dumps(error_content)}\n"

# --- API Endpoints ---
@router.post("/generate")
async def proxy_generate(request: OllamaRequest, settings: Settings = Depends(get_settings)):
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)
    if request.prompt and settings.prompt_prefix and settings.prompt_prefix.strip():
        request_payload_dict["prompt"] = settings.prompt_prefix + request.prompt

    ollama_target_url = f"{settings.ollama_base_url}/api/generate"
    if request.stream:
        return StreamingResponse(_stream_ollama_response(ollama_target_url, request_payload_dict), media_type="application/x-ndjson")
    else:
        return await _ollama_proxy_request("POST", "/api/generate", settings, json_data=request_payload_dict)

@router.get("/tags", response_model=TagsResponse)
async def proxy_tags(settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request("GET", "/api/tags", settings)

@router.post("/show", response_model=ShowResponse)
async def proxy_show(request: ShowRequest, settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request("POST", "/api/show", settings, json_data=request.model_dump(exclude_unset=True, by_alias=True))

@router.post("/chat")
async def proxy_chat(request: ChatRequest, settings: Settings = Depends(get_settings)):
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)
    ollama_target_url = f"{settings.ollama_base_url}/api/chat"
    if request.stream:
        return StreamingResponse(_stream_ollama_response(ollama_target_url, request_payload_dict), media_type="application/x-ndjson")
    else:
        return await _ollama_proxy_request("POST", "/api/chat", settings, json_data=request_payload_dict)

@router.post("/pull")
async def proxy_pull(request: PullRequest, settings: Settings = Depends(get_settings)):
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)
    ollama_target_url = f"{settings.ollama_base_url}/api/pull"
    if request.stream is False:
        return await _ollama_proxy_request("POST", "/api/pull", settings, json_data=request_payload_dict)
    else:
        return StreamingResponse(
            _stream_ollama_response(ollama_target_url, request_payload_dict),
            media_type="application/x-ndjson"
        )

@router.delete("/delete")
async def proxy_delete(request: DeleteRequest, settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request(
        method="DELETE",
        ollama_endpoint="/api/delete",
        settings=settings,
        json_data=request.model_dump(exclude_unset=True, by_alias=True),
        expected_empty_response_status=200
    )

@router.get("/ps", response_model=PsResponse)
async def proxy_ps(settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request(
        method="GET",
        ollama_endpoint="/api/ps",
        settings=settings
    )

@router.get("/version", response_model=VersionResponse)
async def proxy_version(settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request(
        method="GET",
        ollama_endpoint="/api/version",
        settings=settings
    )

@router.post("/copy")
async def proxy_copy(request: CopyRequest, settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request(
        method="POST",
        ollama_endpoint="/api/copy",
        settings=settings,
        json_data=request.model_dump(exclude_unset=True, by_alias=True),
        expected_empty_response_status=200
    )

@router.post("/embed", response_model=EmbedResponse)
async def proxy_embed(request: EmbedRequest, settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request(
        method="POST",
        ollama_endpoint="/api/embed",
        settings=settings,
        json_data=request.model_dump(exclude_unset=True, by_alias=True)
    )

@router.post("/create")
async def proxy_create(request: CreateRequest, settings: Settings = Depends(get_settings)):
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)
    ollama_target_url = f"{settings.ollama_base_url}/api/create"

    if request.stream is False:
        return await _ollama_proxy_request("POST", "/api/create", settings, json_data=request_payload_dict)
    else:
        return StreamingResponse(
            _stream_ollama_response(ollama_target_url, request_payload_dict),
            media_type="application/x-ndjson"
        )
