from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

# --- Existing Models (from /api/generate) ---
class OllamaRequest(BaseModel):
    model: str
    prompt: str # Used by /api/generate
    images: Optional[List[str]] = None # For multimodal models
    format: Optional[str] = None # e.g., "json"
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None # Deprecated but might be sent
    stream: Optional[bool] = False
    raw: Optional[bool] = False
    keep_alive: Optional[Union[str, int]] = None # duration string or int seconds
    suffix: Optional[str] = None


class OllamaResponse(BaseModel): # Primarily for /api/generate
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    done_reason: Optional[str] = None


# --- Models for /api/tags ---
class ModelDetails(BaseModel):
    parent_model: str
    format: str
    family: str
    families: Optional[List[str]] = None
    parameter_size: str
    quantization_level: str

class ModelTagInfo(BaseModel):
    name: str
    model: str # Added field, as API returns both 'name' and 'model' (full name)
    modified_at: str
    size: int
    digest: str
    details: ModelDetails
    # expires_at: Optional[str] = None # Only for /api/ps
    # size_vram: Optional[int] = None  # Only for /api/ps

class TagsResponse(BaseModel):
    models: List[ModelTagInfo]


# --- Models for /api/show ---
class ShowRequest(BaseModel):
    name: str # Corresponds to 'model' in the API doc path, but 'name' in body
    verbose: Optional[bool] = False

class ModelInformation(BaseModel): # For the 'model_info' field in ShowResponse
    # This can have many dynamic keys based on model architecture (e.g. llama, tokenizer)
    # Using Dict[str, Any] to capture this dynamic structure.
    # Specific common fields can be added if needed for validation.
    general_architecture: Optional[str] = Field(None, alias="general.architecture")
    general_file_type: Optional[int] = Field(None, alias="general.file_type")
    # Add other known general fields if necessary
    # For other namespaces like 'llama.*', 'tokenizer.ggml.*', they will be caught by extra='allow' if BaseModel is configured for it
    # or handle them dynamically if needed. For now, keeping it simple.
    class Config:
        extra = "allow" # Allow arbitrary keys for model_info
        populate_by_name = True # Allow using alias for population

class ShowResponse(BaseModel):
    modelfile: Optional[str] = None
    parameters: Optional[str] = None
    template: Optional[str] = None
    details: Optional[ModelDetails] = None # Re-use ModelDetails from TagsResponse
    model_info: Optional[ModelInformation] = None # Detailed model information
    capabilities: Optional[List[str]] = None


# --- Models for /api/chat ---
class ChatMessage(BaseModel):
    role: str  # "system", "user", or "assistant", "tool"
    content: str
    images: Optional[List[str]] = None # List of base64-encoded images
    tool_calls: Optional[List[Dict[str, Any]]] = None # For function calling

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    format: Optional[str] = None # e.g., "json"
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False
    keep_alive: Optional[Union[str, int]] = None
    tools: Optional[List[Dict[str, Any]]] = None

class ChatResponseMessage(BaseModel): # The 'message' field in ChatResponse
    role: str
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel): # For one message in a stream, or the full response if not streaming
    model: str
    created_at: str
    message: Optional[ChatResponseMessage] = None # Optional because final stream object might only have 'done'
    done: bool
    done_reason: Optional[str] = None # e.g. "stop", "length", "load", "error", "unload"
    total_duration: Optional[int] = None # Present if done=true and stream=false
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


# --- Models for /api/pull ---
class PullRequest(BaseModel):
    name: str # Corresponds to 'model' in API doc path, but 'name' in body
    insecure: Optional[bool] = False
    stream: Optional[bool] = True # API doc implies stream is default/common

class PullStatus(BaseModel): # For individual stream objects during pull/push/create
    status: str
    digest: Optional[str] = None
    total: Optional[int] = None
    completed: Optional[int] = None
    error: Optional[str] = None # For error messages in stream


# --- Models for /api/delete ---
class DeleteRequest(BaseModel): # Ollama API takes model name in request body
    name: str # model name to delete (not 'model' field, but 'name')

# --- Models for /api/copy ---
class CopyRequest(BaseModel):
    source: str
    destination: str

# --- Models for /api/embed ---
class EmbedRequest(BaseModel):
    model: str
    input: Union[str, List[str]] # API doc says "text or list of text" but examples use "prompt" for single, "input" for multiple
                                # Official client uses 'input' field. Let's stick to 'input'
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[str, int]] = None
    truncate: Optional[bool] = None


class EmbedResponse(BaseModel):
    # API doc shows "embeddings" for /api/embed, "embedding" for older /api/embeddings
    # Assuming /api/embed structure.
    model: Optional[str] = None # Not in example response, but good practice
    embeddings: List[List[float]]


# --- Models for /api/ps ---
class ProcessModelInfo(ModelTagInfo): # Extends ModelTagInfo with runtime details
    expires_at: Optional[str] = None
    size_vram: Optional[int] = None
    # Parent ModelTagInfo already has: name, model, modified_at, size, digest, details

class PsResponse(BaseModel):
    models: List[ProcessModelInfo]


# --- Models for /api/version ---
class VersionResponse(BaseModel):
    version: str

# --- Models for /api/create ---
class CreateRequest(BaseModel):
    name: str # name of the model to create (becomes 'model' in API)
    from_model: Optional[str] = Field(None, alias="from") # name of existing model
    files: Optional[Dict[str, str]] = None # filename: sha256_digest for GGUF/Safetensors
    adapters: Optional[Dict[str, str]] = None
    template: Optional[str] = None
    license: Optional[Union[str, List[str]]] = None
    system: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    messages: Optional[List[ChatMessage]] = None
    stream: Optional[bool] = True # API doc implies stream is default/common
    quantize: Optional[str] = None

    class Config:
        populate_by_name = True # Allow using alias for population

# Generic response for simple status messages (can be used by copy, delete if they don't have specific bodies)
class StatusOkResponse(BaseModel):
    status: str = "success"
