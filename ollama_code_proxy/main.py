from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from contextlib import asynccontextmanager
import os
import logging
import httpx
import json
from typing import Dict, Any, Optional, Callable, Coroutine, List

from ollama_code_proxy.proxy_server.routes import router as ollama_router
from ollama_code_proxy.code_analyzer.codebase_loader import CodebaseLoader
from ollama_code_proxy.code_analyzer.knowledge_graph import KnowledgeGraph
from ollama_code_proxy.code_analyzer.models import ModuleInfo
from ollama_code_proxy.code_analyzer.context_retriever import ContextRetriever
from ollama_code_proxy.code_analyzer.prompt_program import PromptProgramGenerator
from ollama_code_proxy.code_analyzer.evaluator import CodeEvaluator
from ollama_code_proxy.code_analyzer.prompt_optimizer import PromptOptimizer, OllamaGenerateCallable, PrintLogger

# --- Configuration ---
CODEBASE_ROOT_PATH = os.getenv("CODEBASE_ROOT_PATH")
ENABLE_KG_CONTEXT_STR = os.getenv("ENABLE_KG_CONTEXT", "true").lower()
ENABLE_KG_CONTEXT = ENABLE_KG_CONTEXT_STR == "true"
MAX_CONTEXT_TOKENS_STR = os.getenv("MAX_CONTEXT_TOKENS", "1500")
try: MAX_CONTEXT_TOKENS = int(MAX_CONTEXT_TOKENS_STR)
except ValueError: MAX_CONTEXT_TOKENS = 1500

ENABLE_PROMPT_OPTIMIZER_STR = os.getenv("ENABLE_PROMPT_OPTIMIZER", "true").lower()
ENABLE_PROMPT_OPTIMIZER = ENABLE_PROMPT_OPTIMIZER_STR == "true"
OPTIMIZER_POPULATION_SIZE_STR = os.getenv("OPTIMIZER_POPULATION_SIZE", "5")
OPTIMIZER_NUM_GENERATIONS_STR = os.getenv("OPTIMIZER_NUM_GENERATIONS", "1")
try:
    OPTIMIZER_POPULATION_SIZE = int(OPTIMIZER_POPULATION_SIZE_STR)
    OPTIMIZER_NUM_GENERATIONS = int(OPTIMIZER_NUM_GENERATIONS_STR)
except ValueError:
    OPTIMIZER_POPULATION_SIZE = 5
    OPTIMIZER_NUM_GENERATIONS = 1

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL_FOR_OPTIMIZER = os.getenv("DEFAULT_OLLAMA_MODEL_FOR_OPTIMIZER", "llama3")
OPTIMIZER_LOG_PATH = os.getenv("OPTIMIZER_LOG_PATH")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

feedback_logger_instance: Optional[logging.Logger] = None
if OPTIMIZER_LOG_PATH:
    feedback_logger_instance = logging.getLogger("optimizer_feedback")
    if not feedback_logger_instance.handlers: # Avoid adding multiple handlers on reloads
        feedback_logger_instance.setLevel(logging.INFO)
        # Ensure directory for log path exists
        log_dir = os.path.dirname(OPTIMIZER_LOG_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"Created directory for optimizer log: {log_dir}")

        file_handler = logging.FileHandler(OPTIMIZER_LOG_PATH, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        feedback_logger_instance.addHandler(file_handler)
        feedback_logger_instance.propagate = False
    logger.info(f"Optimizer feedback logging to file enabled. Log file: {OPTIMIZER_LOG_PATH}")
else:
    logger.info("Optimizer feedback logging to file is disabled. Optimizer logs will use standard logger or internal PrintLogger.")

app_state: Dict[str, Any] = {
    "codebase_loader": None, "knowledge_graph": None, "all_modules": None,
    "codebase_root": None, "kg_context_enabled": False,
    "max_context_tokens": MAX_CONTEXT_TOKENS, "analysis_status": "Not initialized",
    "prompt_program_generator": None, "code_evaluator": None,
    "context_retriever_instance": None, "prompt_optimizer": None,
    "optimizer_enabled": False,
    "optimizer_population_size": OPTIMIZER_POPULATION_SIZE,
    "optimizer_num_generations": OPTIMIZER_NUM_GENERATIONS,
    "ollama_base_url": OLLAMA_BASE_URL,
    "feedback_logger": feedback_logger_instance,
    "parsing_error_count": 0,
    "default_ollama_model_for_optimizer": DEFAULT_OLLAMA_MODEL_FOR_OPTIMIZER
}

async def optimizer_ollama_generate_func(
    prompt: str, model: str, stream: bool, options: Optional[Dict[str, Any]]
) -> str:
    payload = {"prompt": prompt, "model": model, "stream": False}
    if options: payload["options"] = options
    api_url = f"{app_state['ollama_base_url']}/api/generate"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "")
    except httpx.HTTPStatusError as e:
        err_resp_text = e.response.text if e.response else str(e)
        logger.error(f"Optimizer's Ollama call failed (HTTP {e.response.status_code if e.response else 'Unknown'}): {err_resp_text}")
        return f"Error: Ollama HTTP {e.response.status_code if e.response else 'Unknown'}. Details: {err_resp_text}"
    except httpx.RequestError as e:
        logger.error(f"Optimizer's Ollama call failed (Request Error): {e}")
        return f"Error: Ollama Connection Error at {api_url}. Detail: {str(e)}" # Corrected str(e)
    except Exception as e:
        logger.error(f"Optimizer's Ollama call failed (Unexpected Error): {e}", exc_info=True)
        return f"Error: Unexpected error during Ollama call. Detail: {str(e)}" # Corrected str(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    # Initialize app_state with values that might be read from env vars again, or defaults
    app_state["max_context_tokens"] = MAX_CONTEXT_TOKENS
    app_state["optimizer_population_size"] = OPTIMIZER_POPULATION_SIZE
    app_state["optimizer_num_generations"] = OPTIMIZER_NUM_GENERATIONS
    app_state["ollama_base_url"] = OLLAMA_BASE_URL
    app_state["default_ollama_model_for_optimizer"] = DEFAULT_OLLAMA_MODEL_FOR_OPTIMIZER
    app_state["feedback_logger"] = feedback_logger_instance

    if not CODEBASE_ROOT_PATH or not os.path.isdir(CODEBASE_ROOT_PATH):
        msg = f"CODEBASE_ROOT_PATH ('{CODEBASE_ROOT_PATH}') not valid. KG context and Optimizer disabled."
        logger.error(msg); app_state.update({"analysis_status": msg, "kg_context_enabled": False, "optimizer_enabled": False})
    elif not ENABLE_KG_CONTEXT:
        msg = "KG context features disabled by ENABLE_KG_CONTEXT. Optimizer also affected if it relies on KG."
        logger.info(msg); app_state.update({"analysis_status": msg, "kg_context_enabled": False})
        if ENABLE_PROMPT_OPTIMIZER: logger.info("Prompt Optimizer effectively disabled as it relies on KG context features.")
        app_state["optimizer_enabled"] = False
    else:
        logger.info(f"Loading/analyzing codebase: {CODEBASE_ROOT_PATH}")
        app_state["analysis_status"] = "Initializing analysis..."
        try:
            abs_codebase_root = os.path.abspath(CODEBASE_ROOT_PATH)
            app_state["codebase_root"] = abs_codebase_root

            def loader_progress(cur: int, total: int, phase: str):
                status_msg = f"Analysis: {phase} - {cur}/{total} ({((cur/total)*100 if total>0 else 0):.1f}%)"
                logger.info(status_msg); app_state["analysis_status"] = status_msg

            loader = CodebaseLoader(abs_codebase_root, progress_callback=loader_progress)
            loader.load_analyze_and_build_graph()

            app_state.update({
                "codebase_loader": loader, "knowledge_graph": loader.get_knowledge_graph(),
                "all_modules": loader.get_all_modules(), "kg_context_enabled": True,
                "parsing_error_count": len(loader.get_parsing_errors())
            })
            status_suffix = f"with {app_state['parsing_error_count']} parsing error(s)" if app_state['parsing_error_count'] > 0 else "successfully"
            app_state["analysis_status"] = f"Analysis complete {status_suffix}. KG context enabled."
            if app_state['parsing_error_count'] > 0: logger.warning(f"{app_state['parsing_error_count']} parsing errors found.")
            logger.info("Codebase analysis finished.")

            if app_state["kg_context_enabled"] and ENABLE_PROMPT_OPTIMIZER:
                logger.info("Initializing Prompt Optimizer components...")
                app_state["prompt_program_generator"] = PromptProgramGenerator()
                app_state["code_evaluator"] = CodeEvaluator()
                context_retriever = ContextRetriever(
                    kg=app_state["knowledge_graph"], all_modules=app_state["all_modules"],
                    codebase_root=app_state["codebase_root"]
                )
                app_state["context_retriever_instance"] = context_retriever

                optimizer_logger_to_use = app_state["feedback_logger"] if app_state["feedback_logger"] else PrintLogger()

                app_state["prompt_optimizer"] = PromptOptimizer(
                    generator=app_state["prompt_program_generator"], evaluator=app_state["code_evaluator"],
                    context_retriever=context_retriever, ollama_generate_func=optimizer_ollama_generate_func,
                    default_ollama_model=app_state["default_ollama_model_for_optimizer"],
                    logger=optimizer_logger_to_use
                )
                app_state["optimizer_enabled"] = True
                logger.info("Prompt Optimizer initialized and enabled.")
            elif ENABLE_PROMPT_OPTIMIZER:
                logger.warning("Prompt Optimizer is enabled in config, but KG context is not available. Optimizer disabled.")
                app_state["optimizer_enabled"] = False
            else:
                 logger.info("Prompt Optimizer is disabled by configuration (ENABLE_PROMPT_OPTIMIZER).")
                 app_state["optimizer_enabled"] = False
        except Exception as e:
            logger.error(f"Failed codebase analysis or optimizer setup: {e}", exc_info=True)
            app_state.update({
                "analysis_status": f"Error: {e}. Features disabled.", "kg_context_enabled": False, "optimizer_enabled": False,
                "knowledge_graph": None, "all_modules": None, "codebase_loader": None,
                "prompt_optimizer": None, "context_retriever_instance": None
            })

    app.state.app_data = app_state
    yield
    logger.info("Application shutdown...")
    current_feedback_logger = app_state.get("feedback_logger")
    if current_feedback_logger and isinstance(current_feedback_logger, logging.Logger):
        for handler in current_feedback_logger.handlers[:]:
            handler.close()
            current_feedback_logger.removeHandler(handler)
    app_state.clear()

app = FastAPI(lifespan=lifespan)
app.include_router(ollama_router, prefix="/api/v1/ollama", tags=["Ollama Proxy"])

@app.get("/api/v1/analysis/status", tags=["Code Analysis Status"])
async def get_analysis_status(request: FastAPIRequest):
    data = request.app.state.app_data
    kg_instance = data.get("knowledge_graph")
    kg_nodes = kg_instance.graph.number_of_nodes() if kg_instance and hasattr(kg_instance, 'graph') else 0
    kg_edges = kg_instance.graph.number_of_edges() if kg_instance and hasattr(kg_instance, 'graph') else 0

    return {
        "analysis_status": data.get("analysis_status", "Unknown"),
        "kg_context_enabled": data.get("kg_context_enabled", False),
        "optimizer_enabled": data.get("optimizer_enabled", False),
        "optimizer_population_size": data.get("optimizer_population_size"),
        "optimizer_num_generations": data.get("optimizer_num_generations"),
        "default_ollama_model_for_optimizer": data.get("default_ollama_model_for_optimizer"),
        "optimizer_log_path": OPTIMIZER_LOG_PATH if OPTIMIZER_LOG_PATH else "Not configured (optimizer logs to console or main logger)",
        "codebase_root_configured": CODEBASE_ROOT_PATH if CODEBASE_ROOT_PATH else "Not Set",
        "codebase_root_used": data.get("codebase_root"),
        "max_context_tokens_configured": data.get("max_context_tokens"),
        "ollama_base_url": data.get("ollama_base_url"), # Renamed key for clarity
        "loaded_modules_count": len(data.get("all_modules", {})) if data.get("all_modules") else 0,
        "knowledge_graph_nodes": kg_nodes, "knowledge_graph_edges": kg_edges,
        "parsing_error_count": data.get("parsing_error_count", 0)
    }

@app.get("/")
async def root(): return {"message": "Ollama Code Proxy is running! See /docs for API details."}
