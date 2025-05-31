from fastapi import APIRouter, HTTPException, Depends, Body, Response, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
import httpx
import os
import json
from typing import Any, Dict, Optional, Union, AsyncGenerator, List as TypingList

from .models import (
    OllamaRequest, OllamaResponse, TagsResponse, ShowRequest, ShowResponse,
    ChatMessage, ChatRequest, ChatResponse, PullRequest, PullStatus,
    DeleteRequest, CopyRequest, EmbedRequest, EmbedResponse, PsResponse,
    VersionResponse, CreateRequest, StatusOkResponse
)
# Corrected relative import path
from ..code_analyzer.context_retriever import ContextRetriever
from ..code_analyzer.knowledge_graph import KnowledgeGraph
from ..code_analyzer.models import ModuleInfo # For type hint in Settings

router = APIRouter() # Prefix is handled in main.py

# --- Configuration Defaults ---
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
# PROMPT_PREFIX_DEFAULT = "Based on your knowledge, please respond to the following: \n" # Simple prefix for /generate if no KG
ENABLE_KG_CONTEXT_DEFAULT = True
MAX_CONTEXT_TOKENS_DEFAULT = 1500

KG_INSTRUCTIONAL_PREFIX = (
    "You are an expert Python programming assistant. The user is asking a question about a codebase.\n"
    "Use the following provided code context to understand the relevant parts of the codebase.\n"
    "The context may include function definitions, class structures, call relationships, and import statements.\n"
    "Based on this context AND the user's request, provide a comprehensive and accurate response.\n"
    "If the context is insufficient, state that and try to answer based on general knowledge if appropriate.\n"
    "Do not refer to 'the context provided' in your answer, just use it."
)

class Settings:
    ollama_base_url: str
    # prompt_prefix: str # Re-evaluating if this is needed globally or per-request type
    enable_kg_context: bool
    max_context_tokens: int
    knowledge_graph: Optional[KnowledgeGraph]
    all_modules: Optional[Dict[str, ModuleInfo]] # Keys are absolute filepaths
    codebase_root: Optional[str] # Absolute path

def get_settings(request: FastAPIRequest) -> Settings:
    app_data = getattr(request.app.state, "app_data", {})

    s = Settings()
    s.ollama_base_url = os.getenv("OLLAMA_BASE_URL", app_data.get("ollama_base_url", OLLAMA_BASE_URL_DEFAULT))
    s.enable_kg_context = app_data.get("kg_context_enabled", ENABLE_KG_CONTEXT_DEFAULT)
    s.max_context_tokens = app_data.get("max_context_tokens", MAX_CONTEXT_TOKENS_DEFAULT)
    s.knowledge_graph = app_data.get("knowledge_graph")
    s.all_modules = app_data.get("all_modules")
    s.codebase_root = app_data.get("codebase_root")
    # s.prompt_prefix = os.getenv("PROMPT_PREFIX", PROMPT_PREFIX_DEFAULT) # Can be added if needed for other endpoints
    return s

async def _ollama_proxy_request(
    method: str, ollama_endpoint: str, settings: Settings,
    json_data: Optional[Dict[str, Any]] = None, params_data: Optional[Dict[str, Any]] = None,
    expected_empty_response_status: Optional[int] = None
) -> Any:
    full_ollama_url = f"{settings.ollama_base_url}{ollama_endpoint}"
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            if method.upper() == "GET": response = await client.get(full_ollama_url, params=params_data)
            elif method.upper() == "POST": response = await client.post(full_ollama_url, json=json_data, params=params_data)
            elif method.upper() == "DELETE": response = await client.request("DELETE", full_ollama_url, json=json_data, params=params_data)
            else: raise HTTPException(status_code=501, detail=f"Unsupported proxy method: {method}")

            response.raise_for_status()
            if expected_empty_response_status and response.status_code == expected_empty_response_status: return Response(status_code=expected_empty_response_status)
            if not response.content and (200 <= response.status_code <= 299): return Response(status_code=response.status_code)
            return response.json()
    except httpx.HTTPStatusError as e:
        error_detail_payload: Any = str(e)
        if e.response and e.response.text:
            try: error_detail_payload = e.response.json()
            except json.JSONDecodeError: error_detail_payload = e.response.text
        raise HTTPException(status_code=e.response.status_code if e.response else 500, detail=error_detail_payload)
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Error connecting to Ollama API at {full_ollama_url}: {str(e)}")
    except Exception as e:
        if isinstance(e, HTTPException): raise
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in proxy request: {str(e)}")

async def _stream_ollama_response(
    ollama_url: str, payload: Dict[str, Any], http_method: str = "POST"
) -> AsyncGenerator[str, None]:
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(http_method, ollama_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines(): yield f"{line}\n"
    except httpx.HTTPStatusError as e:
        error_payload = {"error": "Ollama API error during stream setup", "status_code": e.response.status_code if e.response else 500}
        try: error_payload["detail"] = e.response.json() if e.response else str(e)
        except json.JSONDecodeError: error_payload["detail"] = e.response.text if e.response and e.response.text else str(e)
        except Exception as detail_ex: error_payload["detail"] = str(detail_ex)
        yield f"{json.dumps(error_payload)}\n"
    except httpx.RequestError as e: yield f"{json.dumps({'error': 'Connection to Ollama API failed during stream', 'detail': str(e)})}\n"
    except Exception as e: yield f"{json.dumps({'error': 'Streaming failed unexpectedly', 'detail': str(e)})}\n"

@router.post("/generate")
async def proxy_generate(fastapi_request: FastAPIRequest, request: OllamaRequest, settings: Settings = Depends(get_settings)):
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)
    original_user_prompt = request.prompt
    final_prompt = original_user_prompt

    current_file_context_rel: Optional[str] = None
    if request.options and isinstance(request.options.get("current_file"), str):
        current_file_context_rel = request.options.get("current_file")
        if current_file_context_rel and (".." in current_file_context_rel or os.path.isabs(current_file_context_rel)):
            current_file_context_rel = None

    if settings.enable_kg_context and settings.knowledge_graph and settings.all_modules and settings.codebase_root:
        retriever = ContextRetriever(settings.knowledge_graph, settings.all_modules, settings.codebase_root)
        retrieved_context = retriever.get_context_for_prompt(
            user_prompt=original_user_prompt,
            current_filepath_rel=current_file_context_rel,
            max_context_tokens=settings.max_context_tokens
        )
        if retrieved_context and "No specific code context found" not in retrieved_context :
            final_prompt = (
                f"{KG_INSTRUCTIONAL_PREFIX}\n\n"
                f"### Code Context (from codebase analysis):\n{retrieved_context}\n\n"
                f"### User Request:\n{original_user_prompt}"
            )
    # No simple prefix fallback for /generate if KG context fails, to avoid basic prefixing on complex prompts.

    request_payload_dict["prompt"] = final_prompt

    ollama_target_url = f"{settings.ollama_base_url}/api/generate"
    if request.stream:
        return StreamingResponse(_stream_ollama_response(ollama_target_url, request_payload_dict), media_type="application/x-ndjson")
    else:
        return await _ollama_proxy_request("POST", "/api/generate", settings, json_data=request_payload_dict)

@router.post("/chat")
async def proxy_chat(fastapi_request: FastAPIRequest, request: ChatRequest, settings: Settings = Depends(get_settings)):
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)

    current_file_context_rel: Optional[str] = None
    if request.options and isinstance(request.options.get("current_file"), str):
        current_file_context_rel = request.options.get("current_file")
        if current_file_context_rel and (".." in current_file_context_rel or os.path.isabs(current_file_context_rel)):
            current_file_context_rel = None

    if settings.enable_kg_context and settings.knowledge_graph and settings.all_modules and settings.codebase_root:
        last_user_message_content = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.role == "user":
                    last_user_message_content = msg.content
                    break

        if last_user_message_content:
            retriever = ContextRetriever(settings.knowledge_graph, settings.all_modules, settings.codebase_root)
            retrieved_context = retriever.get_context_for_prompt(
                user_prompt=last_user_message_content,
                current_filepath_rel=current_file_context_rel,
                max_context_tokens=settings.max_context_tokens
            )

            if retrieved_context and "No specific code context found" not in retrieved_context:
                context_system_message_content = (
                    f"{KG_INSTRUCTIONAL_PREFIX}\n\n"
                    f"### Relevant Code Context (from codebase analysis):\n{retrieved_context}"
                )

                messages_list: TypingList[Dict[str, Any]] = request_payload_dict.get("messages", [])
                found_system_message = False
                for i, msg_dict in enumerate(messages_list):
                    if msg_dict.get("role") == "system":
                        messages_list[i]["content"] = context_system_message_content + "\n\n---\n\n" + msg_dict.get("content", "")
                        found_system_message = True
                        break
                if not found_system_message:
                    messages_list.insert(0, {"role": "system", "content": context_system_message_content})
                request_payload_dict["messages"] = messages_list

    ollama_target_url = f"{settings.ollama_base_url}/api/chat"
    if request.stream:
        return StreamingResponse(_stream_ollama_response(ollama_target_url, request_payload_dict), media_type="application/x-ndjson")
    else:
        return await _ollama_proxy_request("POST", "/api/chat", settings, json_data=request_payload_dict)

# Other endpoints
@router.get("/tags", response_model=TagsResponse)
async def proxy_tags(settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request("GET", "/api/tags", settings)

@router.post("/show", response_model=ShowResponse)
async def proxy_show(request: ShowRequest, settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request("POST", "/api/show", settings, json_data=request.model_dump(exclude_unset=True, by_alias=True))

@router.post("/pull")
async def proxy_pull(request: PullRequest, settings: Settings = Depends(get_settings)):
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)
    ollama_target_url = f"{settings.ollama_base_url}/api/pull"
    if request.stream is False:
        return await _ollama_proxy_request("POST", "/api/pull", settings, json_data=request_payload_dict)
    else:
        return StreamingResponse(_stream_ollama_response(ollama_target_url, request_payload_dict), media_type="application/x-ndjson")

@router.delete("/delete")
async def proxy_delete(request: DeleteRequest, settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request(
        method="DELETE", ollama_endpoint="/api/delete", settings=settings,
        json_data=request.model_dump(exclude_unset=True, by_alias=True),
        expected_empty_response_status=200
    )

@router.get("/ps", response_model=PsResponse)
async def proxy_ps(settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request("GET", "/api/ps", settings)

@router.get("/version", response_model=VersionResponse)
async def proxy_version(settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request("GET", "/api/version", settings)

@router.post("/copy")
async def proxy_copy(request: CopyRequest, settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request(
        method="POST", ollama_endpoint="/api/copy", settings=settings,
        json_data=request.model_dump(exclude_unset=True, by_alias=True),
        expected_empty_response_status=200
    )

@router.post("/embed", response_model=EmbedResponse)
async def proxy_embed(request: EmbedRequest, settings: Settings = Depends(get_settings)):
    return await _ollama_proxy_request(
        method="POST", ollama_endpoint="/api/embed", settings=settings,
        json_data=request.model_dump(exclude_unset=True, by_alias=True)
    )

@router.post("/create")
async def proxy_create(request: CreateRequest, settings: Settings = Depends(get_settings)):
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)
    ollama_target_url = f"{settings.ollama_base_url}/api/create"
    if request.stream is False:
        return await _ollama_proxy_request("POST", "/api/create", settings, json_data=request_payload_dict)
    else:
        return StreamingResponse(_stream_ollama_response(ollama_target_url, request_payload_dict), media_type="application/x-ndjson")
