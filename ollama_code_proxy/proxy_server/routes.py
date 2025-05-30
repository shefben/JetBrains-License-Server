from fastapi import APIRouter, HTTPException, Depends
import httpx
import os
from .models import OllamaRequest, OllamaResponse

router = APIRouter()

# Configuration using environment variables with defaults
OLLAMA_API_URL_DEFAULT = "http://localhost:11434/api/generate"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", OLLAMA_API_URL_DEFAULT)

PROMPT_PREFIX_DEFAULT = "Based on your knowledge, please respond to the following: \n"
PROMPT_PREFIX = os.getenv("PROMPT_PREFIX", PROMPT_PREFIX_DEFAULT)

# Dependency to get settings - can be expanded later for more complex configs
class Settings:
    ollama_api_url: str = OLLAMA_API_URL
    prompt_prefix: str = PROMPT_PREFIX

def get_settings():
    return Settings()

@router.post("/generate", response_model=OllamaResponse)
async def proxy_to_ollama(request: OllamaRequest, settings: Settings = Depends(get_settings)):
    # Modify the prompt
    original_prompt = request.prompt
    prefixed_prompt = settings.prompt_prefix + original_prompt

    # Create a new request model or update the existing one for sending
    # Pydantic models are immutable by default, so create a new one or use .copy(update=...)
    request_payload_dict = request.model_dump()
    request_payload_dict["prompt"] = prefixed_prompt

    try:
        async with httpx.AsyncClient() as client:
            # Use the configured API URL
            response = await client.post(settings.ollama_api_url, json=request_payload_dict)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Assuming the response from Ollama matches our OllamaResponse model
            # If stream=True, Ollama's response is a stream of JSON objects,
            # handling that properly would require more complex logic here.
            # For now, this proxy assumes stream=False or that the client handles stream aggregation.
            # The OllamaRequest model has stream: Optional[bool] = False, so this is consistent.
            return response.json()

    except httpx.HTTPStatusError as e:
        # Forward the status code and detail from the Ollama API error
        # Attempt to parse the error response from Ollama if possible
        error_detail = str(e)
        try:
            if e.response and e.response.text:
                # Try to parse as JSON, if not, use raw text
                error_detail = e.response.json()
        except Exception:
            # If response is not JSON or another error occurs, use response text or original exception string
            if e.response and e.response.text:
                error_detail = e.response.text
            # else error_detail remains str(e)
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        # Handle errors connecting to the Ollama API
        raise HTTPException(status_code=503, detail=f"Error connecting to Ollama API at {settings.ollama_api_url}: {str(e)}")
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
