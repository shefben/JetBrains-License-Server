import pytest
import os
from typing import Dict, List, Optional, Any # Added Any
from unittest.mock import MagicMock, patch
import httpx # Not used directly here, but good to have if expanding tests

from ollama_code_proxy.code_analyzer.models import (
    ModuleInfo, FunctionInfo, ClassInfo, ImportInfo, CallInfo, InstanceCreationInfo
)
from ollama_code_proxy.code_analyzer.reference_resolver import ReferenceResolver
from ollama_code_proxy.code_analyzer.knowledge_graph import (
    KnowledgeGraph,
    get_module_node_id,
    get_class_node_id,
    get_function_node_id
)
from fastapi import FastAPI # For the test app
from fastapi.testclient import TestClient # Though not used in these specific integration tests

# --- Mock Data Setup ---

def create_mock_module_info(
    abs_filepath: str, # Expect absolute path for the module itself
    codebase_root: str, # Absolute path to the codebase root for calculating relative path
    module_name: str,
    imports: Optional[List[ImportInfo]] = None,
    functions: Optional[List[FunctionInfo]] = None,
    classes: Optional[List[ClassInfo]] = None,
    language: str = "python",
    docstring: Optional[str] = None,
    variables: Optional[List[Any]] = None,
    parse_errors: Optional[List[Dict[str, Any]]] = None
) -> ModuleInfo:
    # ModuleInfo.filepath should be relative to codebase_root
    rel_filepath = os.path.relpath(abs_filepath, codebase_root)
    return ModuleInfo(
        filepath=rel_filepath,
        module_name=module_name,
        imports=imports or [],
        functions=functions or [],
        classes=classes or [],
        language=language,
        docstring=docstring,
        variables=variables or [],
        parse_errors=parse_errors or []
    )

# --- Pytest Fixtures ---

@pytest.fixture
def codebase_root_path(tmp_path) -> str:
    # Create actual dummy directories for path normalization and os.path.dirname to work
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "pkg"), exist_ok=True)
    # Create empty files so that os.path.exists or similar checks might pass if resolver/KG does them
    # (though current mock setup doesn't require actual file content)
    open(os.path.join(root, "module_a.py"), 'a').close()
    open(os.path.join(root, "pkg", "__init__.py"), 'a').close()
    open(os.path.join(root, "pkg", "module_b.py"), 'a').close()
    return root

@pytest.fixture
def sample_modules_for_resolver(codebase_root_path: str) -> Dict[str, ModuleInfo]:
    # Use absolute paths for dictionary keys (as CodebaseLoader would provide)
    # ModuleInfo.filepath will be relative (as set by create_mock_module_info)

    path_a_abs = os.path.join(codebase_root_path, "module_a.py")
    path_pkg_init_abs = os.path.join(codebase_root_path, "pkg", "__init__.py")
    path_b_abs = os.path.join(codebase_root_path, "pkg", "module_b.py")

    module_a_imports = [
        ImportInfo(module="pkg.module_b", name="func_b", lineno=1, level=0)
    ]
    module_a_calls = [CallInfo(target_name="func_b", lineno=2)]
    module_a_funcs = [FunctionInfo(name="main_a", lineno=2, function_calls=module_a_calls)]
    module_a = create_mock_module_info(path_a_abs, codebase_root_path, "module_a", imports=module_a_imports, functions=module_a_funcs)

    module_b_funcs = [FunctionInfo(name="func_b", lineno=1)] # Definition of func_b
    module_b = create_mock_module_info(path_b_abs, codebase_root_path, "module_b", functions=module_b_funcs)

    pkg_init = create_mock_module_info(path_pkg_init_abs, codebase_root_path, "__init__")

    return {
        path_a_abs: module_a,
        path_b_abs: module_b,
        path_pkg_init_abs: pkg_init
    }

@pytest.fixture
def resolved_sample_modules(sample_modules_for_resolver: Dict[str, ModuleInfo], codebase_root_path: str) -> Dict[str, ModuleInfo]:
    resolver = ReferenceResolver(all_modules=sample_modules_for_resolver, codebase_root=codebase_root_path)
    resolver.resolve_all_references() # Modifies ModuleInfo objects in-place
    return sample_modules_for_resolver

@pytest.fixture
def sample_knowledge_graph(resolved_sample_modules: Dict[str, ModuleInfo], codebase_root_path: str) -> KnowledgeGraph:
    kg = KnowledgeGraph(all_modules=resolved_sample_modules, codebase_root=codebase_root_path)
    return kg

# --- Tests for ReferenceResolver ---

def test_reference_resolver_import_resolution(resolved_sample_modules: Dict[str, ModuleInfo], codebase_root_path: str):
    module_a_abs_path = os.path.join(codebase_root_path, "module_a.py")
    module_a_info = resolved_sample_modules[module_a_abs_path]

    resolved_import = next((imp for imp in module_a_info.imports if imp.name == "func_b"), None)
    assert resolved_import is not None

    # resolved_filepath in ImportInfo should be relative to codebase_root
    expected_rel_path_b = os.path.join("pkg", "module_b.py") # Path relative to root
    # ReferenceResolver stores resolved_filepath relative to codebase_root
    assert resolved_import.resolved_filepath == expected_rel_path_b

def test_reference_resolver_call_resolution(resolved_sample_modules: Dict[str, ModuleInfo], codebase_root_path: str):
    module_a_abs_path = os.path.join(codebase_root_path, "module_a.py")
    module_a_info = resolved_sample_modules[module_a_abs_path]
    main_a_func = module_a_info.functions[0]
    call_to_func_b = main_a_func.function_calls[0]

    expected_rel_path_b = os.path.join("pkg", "module_b.py")
    assert call_to_func_b.resolved_target_filepath == expected_rel_path_b
    assert call_to_func_b.resolved_target_name == "func_b"


# --- Tests for KnowledgeGraph Queries ---

def test_kg_nodes_created(sample_knowledge_graph: KnowledgeGraph, codebase_root_path: str):
    kg = sample_knowledge_graph

    node_module_a_abs = get_module_node_id(os.path.join(codebase_root_path, "module_a.py"))
    node_main_a_abs = get_function_node_id(os.path.join(codebase_root_path, "module_a.py"), "main_a")
    node_module_b_abs = get_module_node_id(os.path.join(codebase_root_path, "pkg", "module_b.py"))
    node_func_b_abs = get_function_node_id(os.path.join(codebase_root_path, "pkg", "module_b.py"), "func_b")

    assert kg.graph.has_node(node_module_a_abs)
    assert kg.get_node_attributes(node_module_a_abs).get("type") == "module"
    assert kg.graph.has_node(node_main_a_abs)
    assert kg.get_node_attributes(node_main_a_abs).get("type") == "function"
    assert kg.graph.has_node(node_module_b_abs)
    assert kg.graph.has_node(node_func_b_abs)


def test_kg_get_callers_and_callees(sample_knowledge_graph: KnowledgeGraph, codebase_root_path: str):
    kg = sample_knowledge_graph

    node_main_a_abs = get_function_node_id(os.path.join(codebase_root_path, "module_a.py"), "main_a")
    node_func_b_abs = get_function_node_id(os.path.join(codebase_root_path, "pkg", "module_b.py"), "func_b")

    callees_of_main_a = kg.get_callees(node_main_a_abs)
    assert node_func_b_abs in callees_of_main_a

    callers_of_func_b = kg.get_callers(node_func_b_abs)
    assert node_main_a_abs in callers_of_func_b

def test_kg_find_definitions(sample_knowledge_graph: KnowledgeGraph, codebase_root_path: str):
    kg = sample_knowledge_graph

    rel_path_b = os.path.join("pkg", "module_b.py") # find_definitions expects relative path
    abs_path_b = os.path.join(codebase_root_path, rel_path_b)

    func_b_defs = kg.find_definitions("func_b", module_filepath_rel=rel_path_b)
    assert len(func_b_defs) == 1
    assert func_b_defs[0] == get_function_node_id(abs_path_b, "func_b")

    func_b_defs_global = kg.find_definitions("func_b")
    assert get_function_node_id(abs_path_b, "func_b") in func_b_defs_global


def test_kg_find_references_to_node(sample_knowledge_graph: KnowledgeGraph, codebase_root_path: str):
    kg = sample_knowledge_graph

    node_func_b_abs = get_function_node_id(os.path.join(codebase_root_path, "pkg", "module_b.py"), "func_b")
    node_main_a_abs = get_function_node_id(os.path.join(codebase_root_path, "module_a.py"), "main_a")

    refs_to_func_b = kg.find_references_to_node(node_func_b_abs)
    assert "called_by" in refs_to_func_b
    assert node_main_a_abs in refs_to_func_b["called_by"]

# --- Test for Root Endpoint (if main app was used, ensures it's not available on this test app) ---
# This test is mostly for sanity checking the test app setup.
def test_main_app_root_not_on_integration_test_app():
    # Create a minimal app for testing router logic, similar to how it's done in test_proxy_server.py
    # This app does NOT have the global "/" endpoint defined in the actual main.py
    test_specific_app = FastAPI()
    test_specific_app.include_router(proxy_routes_module.router, prefix="/api/v1/ollama")

    temp_client = TestClient(test_specific_app)
    response = temp_client.get("/")
    assert response.status_code == 404 # Expect 404 because "/" is not defined on this specific test app
