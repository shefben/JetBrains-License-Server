from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
import os
import logging
from typing import Dict, Any, Optional

# Assuming these are the correct import paths based on project structure
from ollama_code_proxy.proxy_server.routes import router as ollama_router
from ollama_code_proxy.code_analyzer.codebase_loader import CodebaseLoader
from ollama_code_proxy.code_analyzer.knowledge_graph import KnowledgeGraph
from ollama_code_proxy.code_analyzer.models import ModuleInfo


# --- Configuration ---
CODEBASE_ROOT_PATH = os.getenv("CODEBASE_ROOT_PATH")
ENABLE_KG_CONTEXT_STR = os.getenv("ENABLE_KG_CONTEXT", "true").lower()
ENABLE_KG_CONTEXT = ENABLE_KG_CONTEXT_STR == "true"
MAX_CONTEXT_TOKENS_STR = os.getenv("MAX_CONTEXT_TOKENS", "1500") # Adjusted default from script
try:
    MAX_CONTEXT_TOKENS = int(MAX_CONTEXT_TOKENS_STR)
except ValueError:
    MAX_CONTEXT_TOKENS = 1500

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Application State Data ---
app_state: Dict[str, Any] = {
    "codebase_loader": None,
    "knowledge_graph": None,
    "all_modules": None,
    "codebase_root": None,
    "kg_context_enabled": False,
    "max_context_tokens": MAX_CONTEXT_TOKENS, # Store the actual value used
    "analysis_status": "Not initialized",
    "parsing_error_count": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup sequence initiated...")

    app_state["max_context_tokens"] = MAX_CONTEXT_TOKENS # Store configured value

    if not CODEBASE_ROOT_PATH or not os.path.isdir(CODEBASE_ROOT_PATH):
        logger.error(f"CODEBASE_ROOT_PATH ('{CODEBASE_ROOT_PATH}') is not set or is not a valid directory.")
        logger.warning("Knowledge Graph context features will be disabled.")
        app_state["analysis_status"] = f"Error: CODEBASE_ROOT_PATH ('{CODEBASE_ROOT_PATH}') not valid. Context features disabled."
        app_state["kg_context_enabled"] = False
    elif not ENABLE_KG_CONTEXT:
        logger.info("Knowledge Graph context features are disabled by configuration (ENABLE_KG_CONTEXT=false).")
        app_state["analysis_status"] = "Context features disabled by configuration (set ENABLE_KG_CONTEXT=true)."
        app_state["kg_context_enabled"] = False
    else:
        logger.info(f"Attempting to load and analyze codebase at: {CODEBASE_ROOT_PATH}")
        app_state["analysis_status"] = "Initializing codebase analysis..."
        try:
            abs_codebase_root = os.path.abspath(CODEBASE_ROOT_PATH)
            app_state["codebase_root"] = abs_codebase_root

            def loader_progress(current: int, total: int, phase_msg: str):
                progress_percent = (current / total) * 100 if total > 0 else 0
                status_msg = f"In progress: {phase_msg} ({current}/{total} - {progress_percent:.2f}%)"
                logger.info(status_msg)
                app_state["analysis_status"] = status_msg

            loader = CodebaseLoader(codebase_root=abs_codebase_root, progress_callback=loader_progress)
            app_state["codebase_loader"] = loader

            loader.load_analyze_and_build_graph()

            app_state["knowledge_graph"] = loader.get_knowledge_graph()
            app_state["all_modules"] = loader.get_all_modules()
            app_state["kg_context_enabled"] = True

            parsing_errors = loader.get_parsing_errors()
            app_state["parsing_error_count"] = len(parsing_errors)
            final_status_msg = "Analysis complete."
            if parsing_errors:
                 logger.warning(f"Encountered {len(parsing_errors)} parsing error(s) during codebase analysis.")
                 final_status_msg += f" Encountered {len(parsing_errors)} parsing error(s)."

            app_state["analysis_status"] = f"{final_status_msg} KG context enabled."
            logger.info("Codebase analysis finished. Knowledge Graph context is enabled.")

        except Exception as e:
            logger.error(f"Failed to initialize codebase analysis: {e}", exc_info=True)
            app_state["analysis_status"] = f"Error during analysis: {str(e)}. Context features disabled."
            app_state["kg_context_enabled"] = False
            app_state["knowledge_graph"] = None
            app_state["all_modules"] = None

    app.state.app_data = app_state # Make state accessible
    yield
    logger.info("Application shutdown sequence initiated...")
    app_state.clear()


app = FastAPI(lifespan=lifespan)

app.include_router(ollama_router, prefix="/api/v1/ollama", tags=["Ollama Proxy"])

@app.get("/api/v1/analysis/status", tags=["Code Analysis Status"])
async def get_analysis_status(request: Request):
    data = request.app.state.app_data

    kg_instance: Optional[KnowledgeGraph] = data.get("knowledge_graph") # Type hint for clarity
    kg_nodes = kg_instance.graph.number_of_nodes() if kg_instance and kg_instance.graph else 0
    kg_edges = kg_instance.graph.number_of_edges() if kg_instance and kg_instance.graph else 0

    return {
        "analysis_status": data.get("analysis_status", "Unknown"),
        "kg_context_enabled": data.get("kg_context_enabled", False),
        "codebase_root_configured": CODEBASE_ROOT_PATH if CODEBASE_ROOT_PATH else "Not Set",
        "codebase_root_used": data.get("codebase_root"),
        "max_context_tokens_configured": data.get("max_context_tokens"),
        "loaded_modules_count": len(data.get("all_modules", {})) if data.get("all_modules") else 0,
        "knowledge_graph_nodes": kg_nodes,
        "knowledge_graph_edges": kg_edges,
        "parsing_error_count": data.get("parsing_error_count", 0)
    }

@app.get("/")
async def root():
    return {"message": "Ollama Code Proxy is running! See /docs for API details."}

# if __name__ == "__main__":
#     import uvicorn
#     # Example: export CODEBASE_ROOT_PATH=$(pwd)/ollama_code_proxy
#     if not CODEBASE_ROOT_PATH:
#         logger.warning("CODEBASE_ROOT_PATH environment variable not set for local run.")
#         logger.warning("Context-aware features will be disabled or may fail.")
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
