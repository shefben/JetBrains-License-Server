from fastapi import APIRouter, HTTPException, Depends, Body, Response, Request as FastAPIRequest, Query
from fastapi.responses import StreamingResponse
import httpx
import os
import json
import asyncio
import time
from typing import Any, Dict, Optional, Union, AsyncGenerator, List as TypingList

from .models import (
    OllamaRequest, OllamaResponse, TagsResponse, ShowRequest, ShowResponse,
    ChatMessage, ChatRequest, ChatResponse, PullRequest, PullStatus,
    DeleteRequest, CopyRequest, EmbedRequest, EmbedResponse, PsResponse,
    VersionResponse, CreateRequest, StatusOkResponse
    # ModuleInfo is not directly used in routes.py, but by dependencies.
    # It's imported in main.py where app_state['all_modules'] is populated.
)
from ..code_analyzer.context_retriever import ContextRetriever
from ..code_analyzer.knowledge_graph import KnowledgeGraph # For type hint in Settings
from ..code_analyzer.prompt_optimizer import PromptOptimizer
from ..code_analyzer.prompt_program import PromptProgram # For type hint in Settings
from ..code_analyzer.models import ModuleInfo # For type hint in Settings

router = APIRouter() # Prefix is handled in main.py

# --- Configuration Defaults (primarily for get_settings if app_state isn't fully populated) ---
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
# PROMPT_PREFIX_DEFAULT = "..." # This is now less relevant for generate/chat with KG/Optimizer
ENABLE_KG_CONTEXT_DEFAULT = True
MAX_CONTEXT_TOKENS_DEFAULT = 1500
ENABLE_PROMPT_OPTIMIZER_DEFAULT = True
OPTIMIZER_POPULATION_SIZE_DEFAULT = 5
OPTIMIZER_NUM_GENERATIONS_DEFAULT = 1


KG_INSTRUCTIONAL_PREFIX = (
    "You are an expert Python programming assistant. The user is asking a question about a codebase.\n"
    "Use the following provided code context to understand the relevant parts of the codebase.\n"
    "The context may include function definitions, class structures, call relationships, and import statements.\n"
    "Based on this context AND the user's request, provide a comprehensive and accurate response.\n"
    "If the context is insufficient, state that and try to answer based on general knowledge if appropriate.\n"
    "Do not refer to 'the context provided' in your answer, just use it."
)

class Settings: # Used by get_settings dependency
    ollama_base_url: str
    # prompt_prefix: str # Not directly used by generate/chat if KG/Optimizer active
    enable_kg_context: bool
    max_context_tokens: int
    knowledge_graph: Optional[KnowledgeGraph]
    all_modules: Optional[Dict[str, ModuleInfo]] # Keys: abs_filepath, Val: ModuleInfo
    codebase_root: Optional[str] # Absolute path
    optimizer_enabled: bool
    prompt_optimizer: Optional[PromptOptimizer]
    optimizer_population_size: int
    optimizer_num_generations: int

def get_settings(request: FastAPIRequest) -> Settings:
    app_data = getattr(request.app.state, "app_data", {})
    s = Settings()
    s.ollama_base_url = app_data.get("ollama_base_url", OLLAMA_BASE_URL_DEFAULT) # Get from app_state, then default
    s.enable_kg_context = app_data.get("kg_context_enabled", ENABLE_KG_CONTEXT_DEFAULT)
    s.max_context_tokens = app_data.get("max_context_tokens", MAX_CONTEXT_TOKENS_DEFAULT)
    s.knowledge_graph = app_data.get("knowledge_graph")
    s.all_modules = app_data.get("all_modules")
    s.codebase_root = app_data.get("codebase_root")
    s.optimizer_enabled = app_data.get("optimizer_enabled", ENABLE_PROMPT_OPTIMIZER_DEFAULT)
    s.prompt_optimizer = app_data.get("prompt_optimizer")
    s.optimizer_population_size = app_data.get("optimizer_population_size", OPTIMIZER_POPULATION_SIZE_DEFAULT)
    s.optimizer_num_generations = app_data.get("optimizer_num_generations", OPTIMIZER_NUM_GENERATIONS_DEFAULT)
    return s

async def _ollama_proxy_request(
    method: str, ollama_endpoint: str, settings: Settings,
    json_data: Optional[Dict[str, Any]] = None, params_data: Optional[Dict[str, Any]] = None,
    expected_empty_response_status: Optional[int] = None
) -> Any:
    full_ollama_url = f"{settings.ollama_base_url}{ollama_endpoint}"
    try:
        async with httpx.AsyncClient(timeout=None) as client: # Default timeout is 5s, often too short for LLMs
            if method.upper() == "GET": response = await client.get(full_ollama_url, params=params_data)
            elif method.upper() == "POST": response = await client.post(full_ollama_url, json=json_data, params=params_data)
            elif method.upper() == "DELETE": response = await client.request("DELETE", full_ollama_url, json=json_data, params=params_data)
            else: raise HTTPException(status_code=501, detail=f"Unsupported proxy method: {method}")
            response.raise_for_status()
            if expected_empty_response_status and response.status_code == expected_empty_response_status: return Response(status_code=expected_empty_response_status)
            if not response.content and (200 <= response.status_code <= 299): return Response(status_code=response.status_code)
            return response.json()
    except httpx.HTTPStatusError as e:
        error_detail: Any = str(e)
        if e.response and e.response.text:
            try: error_detail = e.response.json()
            except json.JSONDecodeError: error_detail = e.response.text
        raise HTTPException(status_code=e.response.status_code if e.response else 500, detail=error_detail)
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
async def proxy_generate(
    fastapi_request: FastAPIRequest,
    request: OllamaRequest,
    settings: Settings = Depends(get_settings),
    optimize_prompt: bool = Query(False, description="Enable experimental prompt optimization.")
):
    original_user_prompt = request.prompt

    if optimize_prompt and settings.optimizer_enabled and settings.prompt_optimizer:
        current_file_context_rel: Optional[str] = None
        if request.options and isinstance(request.options.get("current_file"), str):
            current_file_context_rel = request.options.get("current_file")
            if current_file_context_rel and (".." in current_file_context_rel or os.path.isabs(current_file_context_rel)):
                current_file_context_rel = None

        abs_file_context: Optional[str] = None
        if current_file_context_rel and settings.codebase_root:
            abs_file_context = os.path.normpath(os.path.join(settings.codebase_root, current_file_context_rel))
            if not abs_file_context.startswith(settings.codebase_root): abs_file_context = None

        optimizer_result = await settings.prompt_optimizer.find_best_prompt_and_response(
            user_request=original_user_prompt, current_filepath_abs=abs_file_context,
            population_size=settings.optimizer_population_size, num_generations=settings.optimizer_num_generations
        )
        if optimizer_result and optimizer_result.generated_code:
            return OllamaResponse(
                model=settings.prompt_optimizer.default_ollama_model,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                response=optimizer_result.generated_code, done=True
            )
        else: raise HTTPException(status_code=500, detail="Prompt optimization failed to produce a result.")

    # Standard KG Context Logic
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)
    final_prompt = original_user_prompt

    current_file_rel_for_kg: Optional[str] = None
    if request.options and isinstance(request.options.get("current_file"), str):
        current_file_rel_for_kg = request.options.get("current_file")
        if current_file_rel_for_kg and (".." in current_file_rel_for_kg or os.path.isabs(current_file_rel_for_kg)):
            current_file_rel_for_kg = None

    if settings.enable_kg_context and settings.knowledge_graph and settings.all_modules and settings.codebase_root:
        retriever = ContextRetriever(settings.knowledge_graph, settings.all_modules, settings.codebase_root)
        abs_file_for_kg_context: Optional[str] = None
        if current_file_rel_for_kg:
             abs_file_for_kg_context = os.path.normpath(os.path.join(settings.codebase_root, current_file_rel_for_kg))
             if not abs_file_for_kg_context.startswith(settings.codebase_root): abs_file_for_kg_context = None

        retrieved_context = retriever.get_context_for_prompt(
            user_prompt=original_user_prompt, current_filepath_abs=abs_file_for_kg_context,
            max_context_tokens=settings.max_context_tokens
        )
        if retrieved_context and "No specific code context found" not in retrieved_context :
            final_prompt = (f"{KG_INSTRUCTIONAL_PREFIX}\n\n### Code Context (from codebase analysis):\n{retrieved_context}\n\n### User Request:\n{original_user_prompt}")

    request_payload_dict["prompt"] = final_prompt
    ollama_target_url = f"{settings.ollama_base_url}/api/generate"
    if request.stream:
        return StreamingResponse(_stream_ollama_response(ollama_target_url, request_payload_dict), media_type="application/x-ndjson")
    else:
        return await _ollama_proxy_request("POST", "/api/generate", settings, json_data=request_payload_dict)

@router.post("/chat")
async def proxy_chat(
    fastapi_request: FastAPIRequest, request: ChatRequest, settings: Settings = Depends(get_settings),
    optimize_prompt: bool = Query(False, description="Enable experimental prompt optimization for the last user message.")
):
    if optimize_prompt and settings.optimizer_enabled and settings.prompt_optimizer:
        last_user_message_content = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.role == "user": last_user_message_content = msg.content; break
        if not last_user_message_content: raise HTTPException(status_code=400, detail="Cannot optimize chat: No user message found.")

        current_file_context_rel_for_opt: Optional[str] = None
        if request.options and isinstance(request.options.get("current_file"), str):
            current_file_context_rel_for_opt = request.options.get("current_file")
            if current_file_context_rel_for_opt and (".." in current_file_context_rel_for_opt or os.path.isabs(current_file_context_rel_for_opt)):
                current_file_context_rel_for_opt = None

        abs_file_context_for_opt: Optional[str] = None
        if current_file_context_rel_for_opt and settings.codebase_root:
             abs_file_context_for_opt = os.path.normpath(os.path.join(settings.codebase_root, current_file_context_rel_for_opt))
             if not abs_file_context_for_opt.startswith(settings.codebase_root): abs_file_context_for_opt = None

        optimizer_result = await settings.prompt_optimizer.find_best_prompt_and_response(
            user_request=last_user_message_content, current_filepath_abs=abs_file_context_for_opt,
            population_size=settings.optimizer_population_size, num_generations=settings.optimizer_num_generations
        )
        if optimizer_result and optimizer_result.generated_code:
            return ChatResponse(
                model=settings.prompt_optimizer.default_ollama_model, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                message=ChatMessage(role="assistant", content=optimizer_result.generated_code), done=True
            )
        else: raise HTTPException(status_code=500, detail="Prompt optimization for chat failed.")

    # Standard KG Context Logic
    request_payload_dict = request.model_dump(exclude_unset=True, by_alias=True)
    current_file_rel_for_kg_chat: Optional[str] = None
    if request.options and isinstance(request.options.get("current_file"), str):
        current_file_rel_for_kg_chat = request.options.get("current_file")
        if current_file_rel_for_kg_chat and (".." in current_file_rel_for_kg_chat or os.path.isabs(current_file_rel_for_kg_chat)):
             current_file_rel_for_kg_chat = None

    if settings.enable_kg_context and settings.knowledge_graph and settings.all_modules and settings.codebase_root:
        last_user_msg_content_for_kg = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.role == "user": last_user_msg_content_for_kg = msg.content; break

        if last_user_msg_content_for_kg:
            retriever = ContextRetriever(settings.knowledge_graph, settings.all_modules, settings.codebase_root)
            abs_file_for_kg_chat: Optional[str] = None
            if current_file_rel_for_kg_chat:
                abs_file_for_kg_chat = os.path.normpath(os.path.join(settings.codebase_root, current_file_rel_for_kg_chat))
                if not abs_file_for_kg_chat.startswith(settings.codebase_root): abs_file_for_kg_chat = None

            retrieved_context = retriever.get_context_for_prompt(
                user_prompt=last_user_msg_content_for_kg, current_filepath_abs=abs_file_for_kg_chat,
                max_context_tokens=settings.max_context_tokens
            )
            if retrieved_context and "No specific code context found" not in retrieved_context:
                context_system_message = ChatMessage(role="system", content=(
                    f"{KG_INSTRUCTIONAL_PREFIX}\n\n### Relevant Code Context (from codebase analysis):\n{retrieved_context}"
                ))
                # Convert messages to list of dicts for modification
                messages_as_dicts: TypingList[Dict[str,Any]] = [m.model_dump(exclude_unset=True) for m in request.messages]

                found_system_message = False
                for i, msg_dict in enumerate(messages_as_dicts):
                    if msg_dict.get("role") == "system":
                        messages_as_dicts[i]["content"] = context_system_message.content + "\n\n---\n\n" + msg_dict.get("content", "")
                        found_system_message = True; break
                if not found_system_message:
                    messages_as_dicts.insert(0, context_system_message.model_dump(exclude_unset=True))
                request_payload_dict["messages"] = messages_as_dicts

    ollama_target_url = f"{settings.ollama_base_url}/api/chat"
    if request.stream:
        return StreamingResponse(_stream_ollama_response(ollama_target_url, request_payload_dict), media_type="application/x-ndjson")
    else:
        return await _ollama_proxy_request("POST", "/api/chat", settings, json_data=request_payload_dict)

@router.get("/tags", response_model=TagsResponse)
async def proxy_tags(settings: Settings = Depends(get_settings)): return await _ollama_proxy_request("GET", "/api/tags", settings)
@router.post("/show", response_model=ShowResponse)
async def proxy_show(request: ShowRequest, settings: Settings = Depends(get_settings)): return await _ollama_proxy_request("POST", "/api/show", settings, json_data=request.model_dump(exclude_unset=True, by_alias=True))
@router.post("/pull")
async def proxy_pull(request: PullRequest, settings: Settings = Depends(get_settings)):
    payload = request.model_dump(exclude_unset=True, by_alias=True)
    url = f"{settings.ollama_base_url}/api/pull"
    if request.stream is False: return await _ollama_proxy_request("POST", "/api/pull", settings, json_data=payload)
    else: return StreamingResponse(_stream_ollama_response(url, payload), media_type="application/x-ndjson")
@router.delete("/delete")
async def proxy_delete(request: DeleteRequest, settings: Settings = Depends(get_settings)): return await _ollama_proxy_request(method="DELETE", ollama_endpoint="/api/delete", settings=settings, json_data=request.model_dump(exclude_unset=True, by_alias=True), expected_empty_response_status=200)
@router.get("/ps", response_model=PsResponse)
async def proxy_ps(settings: Settings = Depends(get_settings)): return await _ollama_proxy_request("GET", "/api/ps", settings)
@router.get("/version", response_model=VersionResponse)
async def proxy_version(settings: Settings = Depends(get_settings)): return await _ollama_proxy_request("GET", "/api/version", settings)
@router.post("/copy")
async def proxy_copy(request: CopyRequest, settings: Settings = Depends(get_settings)): return await _ollama_proxy_request(method="POST", ollama_endpoint="/api/copy", settings=settings, json_data=request.model_dump(exclude_unset=True, by_alias=True), expected_empty_response_status=200)
@router.post("/embed", response_model=EmbedResponse)
async def proxy_embed(request: EmbedRequest, settings: Settings = Depends(get_settings)): return await _ollama_proxy_request("POST", "/api/embed", settings, json_data=request.model_dump(exclude_unset=True, by_alias=True))
@router.post("/create")
async def proxy_create(request: CreateRequest, settings: Settings = Depends(get_settings)):
    payload = request.model_dump(exclude_unset=True, by_alias=True)
    url = f"{settings.ollama_base_url}/api/create"
    if request.stream is False: return await _ollama_proxy_request("POST", "/api/create", settings, json_data=payload)
    else: return StreamingResponse(_stream_ollama_response(url, payload), media_type="application/x-ndjson")
