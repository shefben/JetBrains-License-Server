import pytest
import os
from typing import Dict, List, Optional, Any, Tuple # Added Tuple, Any
from unittest.mock import MagicMock, patch
import httpx # Not used directly here, but good to have if expanding tests

from ollama_code_proxy.code_analyzer.models import (
    ModuleInfo, FunctionInfo, ClassInfo, ImportInfo, CallInfo, InstanceCreationInfo,
    ArgumentInfo, DecoratorInfo, CallArgumentInfo # Added missing imports
)
from ollama_code_proxy.code_analyzer.reference_resolver import ReferenceResolver
from ollama_code_proxy.code_analyzer.knowledge_graph import (
    KnowledgeGraph,
    get_module_node_id,
    get_class_node_id,
    get_function_node_id,
    get_method_node_id # Added missing import
)
from fastapi import FastAPI, Response as FastAPIResponse # For the test app & mock response
from fastapi.testclient import TestClient
# Import proxy_routes_module for the root endpoint test, assuming it's needed.
# However, test_code_analysis_integration.py should focus on code_analyzer components.
# The test_main_app_root_not_on_integration_test_app was trying to use proxy_routes_module
# which is for Ollama proxy routes, not relevant here.
# I will remove that specific test as it's out of scope for this file.

# --- Mock Data Setup ---

def create_mock_module_info(
    abs_filepath: str,
    codebase_root: str,
    module_name: str,
    imports: Optional[List[ImportInfo]] = None,
    functions: Optional[List[FunctionInfo]] = None,
    classes: Optional[List[ClassInfo]] = None,
    language: str = "python",
    docstring: Optional[str] = None,
    variables: Optional[List[Any]] = None,
    parse_errors: Optional[List[Dict[str, Any]]] = None
) -> ModuleInfo:
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
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "pkg"), exist_ok=True)
    os.makedirs(os.path.join(root, "utils"), exist_ok=True) # For ContextRetriever tests

    # Create empty files that will be written to by specific test fixtures if needed
    # This ensures paths are valid for KG node ID generation even if file content is mocked later.
    open(os.path.join(root, "module_a.py"), 'a').close()
    open(os.path.join(root, "pkg", "__init__.py"), 'a').close()
    open(os.path.join(root, "pkg", "module_b.py"), 'a').close()
    open(os.path.join(root, "utils", "helpers.py"), 'a').close()
    return root

@pytest.fixture
def sample_modules_for_resolver(codebase_root_path: str) -> Dict[str, ModuleInfo]:
    path_a_abs = os.path.join(codebase_root_path, "module_a.py")
    path_pkg_init_abs = os.path.join(codebase_root_path, "pkg", "__init__.py")
    path_b_abs = os.path.join(codebase_root_path, "pkg", "module_b.py")

    module_a_imports = [
        ImportInfo(module="pkg.module_b", name="func_b", lineno=1, level=0)
    ]
    module_a_calls = [CallInfo(target_name="func_b", lineno=2, args=[], keywords=[])]
    module_a_funcs = [FunctionInfo(name="main_a", lineno=2, function_calls=module_a_calls, args=[], decorators=[],
                                   instance_creations=[], attribute_accesses=[])]
    module_a = create_mock_module_info(path_a_abs, codebase_root_path, "module_a", imports=module_a_imports, functions=module_a_funcs)

    module_b_funcs = [FunctionInfo(name="func_b", lineno=1, args=[], decorators=[], function_calls=[],
                                   instance_creations=[], attribute_accesses=[])]
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
    resolver.resolve_all_references()
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
    expected_rel_path_b = os.path.join("pkg", "module_b.py")
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

    assert kg.graph.has_node(node_module_a_abs); assert kg.get_node_attributes(node_module_a_abs).get("type") == "module"
    assert kg.graph.has_node(node_main_a_abs); assert kg.get_node_attributes(node_main_a_abs).get("type") == "function"
    assert kg.graph.has_node(node_module_b_abs); assert kg.get_node_attributes(node_module_b_abs).get("type") == "module"
    assert kg.graph.has_node(node_func_b_abs); assert kg.get_node_attributes(node_func_b_abs).get("type") == "function"

def test_kg_get_callers_and_callees(sample_knowledge_graph: KnowledgeGraph, codebase_root_path: str):
    kg = sample_knowledge_graph
    node_main_a_abs = get_function_node_id(os.path.join(codebase_root_path, "module_a.py"), "main_a")
    node_func_b_abs = get_function_node_id(os.path.join(codebase_root_path, "pkg", "module_b.py"), "func_b")
    assert node_func_b_abs in kg.get_callees(node_main_a_abs)
    assert node_main_a_abs in kg.get_callers(node_func_b_abs)

def test_kg_find_definitions(sample_knowledge_graph: KnowledgeGraph, codebase_root_path: str):
    kg = sample_knowledge_graph
    rel_path_b = os.path.join("pkg", "module_b.py")
    abs_path_b = os.path.join(codebase_root_path, rel_path_b)
    func_b_defs = kg.find_definitions("func_b", module_filepath_rel=rel_path_b)
    assert len(func_b_defs) == 1 and func_b_defs[0] == get_function_node_id(abs_path_b, "func_b")
    assert get_function_node_id(abs_path_b, "func_b") in kg.find_definitions("func_b")

def test_kg_find_references_to_node(sample_knowledge_graph: KnowledgeGraph, codebase_root_path: str):
    kg = sample_knowledge_graph
    node_func_b_abs = get_function_node_id(os.path.join(codebase_root_path, "pkg", "module_b.py"), "func_b")
    node_main_a_abs = get_function_node_id(os.path.join(codebase_root_path, "module_a.py"), "main_a")
    refs_to_func_b = kg.find_references_to_node(node_func_b_abs)
    assert "called_by" in refs_to_func_b and node_main_a_abs in refs_to_func_b["called_by"]

# --- Tests for ContextRetriever ---
from ollama_code_proxy.code_analyzer.context_retriever import ContextRetriever, get_source_snippet

@pytest.fixture
def retriever_fixture_data(codebase_root_path: str) -> Tuple[KnowledgeGraph, Dict[str, ModuleInfo], str]:
    utils_helpers_abs_path = os.path.join(codebase_root_path, "utils", "helpers.py")
    utils_helpers_content = """# Line 1
class StringHelper: # Line 2
    '''Helper for strings.''' # Line 3
    def capitalize_first(self, text: str) -> str: # Line 4
        if not text: return "" # Line 5
        return text[0].upper() + text[1:] # Line 6

# Line 7
def format_user(name: str, age: int) -> str: # Line 8
    '''Formats user info.''' # Line 9
    sh = StringHelper() # Line 10
    cap_name = sh.capitalize_first(name) # Line 11
    return f"{cap_name} is {age} years old." # Line 12
"""
    with open(utils_helpers_abs_path, "w") as f: f.write(utils_helpers_content)

    sh_capitalize_method = MethodInfo(name="capitalize_first", lineno=4, end_lineno=6, args=[ArgumentInfo(name="self"), ArgumentInfo(name="text", annotation="str")], returns="str", docstring="", function_calls=[], instance_creations=[], attribute_accesses=[], decorators=[], is_async=False, is_static=False, is_classmethod=False)
    string_helper_class = ClassInfo(name="StringHelper", lineno=2, end_lineno=6, methods=[sh_capitalize_method], docstring="Helper for strings.", bases=[], class_variables=[], decorators=[])

    format_user_inst_creations = [InstanceCreationInfo(class_name="StringHelper", lineno=10, args=[], keywords=[])]
    format_user_calls = [CallInfo(target_name="sh.capitalize_first", lineno=11, args=[CallArgumentInfo(value_repr="name")], keywords=[])]
    format_user_func = FunctionInfo(name="format_user", lineno=8, end_lineno=12, args=[ArgumentInfo(name="name", annotation="str"), ArgumentInfo(name="age", annotation="int")], returns="str", docstring="Formats user info.", instance_creations=format_user_inst_creations, function_calls=format_user_calls, attribute_accesses=[], decorators=[], is_async=False)

    helpers_module = create_mock_module_info(filepath=utils_helpers_abs_path, codebase_root=codebase_root_path, module_name="helpers", classes=[string_helper_class], functions=[format_user_func], docstring="# Line 1")
    all_modules_data = {utils_helpers_abs_path: helpers_module}

    # Manually resolve for the test (as ReferenceResolver is not run here directly)
    helpers_module.functions[0].instance_creations[0].resolved_target_filepath = os.path.relpath(utils_helpers_abs_path, codebase_root_path)
    call_in_format_user = helpers_module.functions[0].function_calls[0]
    call_in_format_user.resolved_target_filepath = os.path.relpath(utils_helpers_abs_path, codebase_root_path)
    call_in_format_user.resolved_target_name = "StringHelper.capitalize_first"
    call_in_format_user.is_method_call = True

    kg = KnowledgeGraph(all_modules_data, codebase_root_path)
    # Manually add call edge for KG, as resolver isn't run on this specific mock data for KG
    caller_node = get_function_node_id(utils_helpers_abs_path, "format_user")
    callee_node = get_method_node_id(utils_helpers_abs_path, "StringHelper", "capitalize_first")
    if kg.graph.has_node(caller_node) and kg.graph.has_node(callee_node):
        kg.graph.add_edge(caller_node, callee_node, type="calls", lineno=11)

    return kg, all_modules_data, codebase_root_path

def test_get_source_snippet_valid(codebase_root_path):
    test_file = os.path.join(codebase_root_path, "test_snippet.py")
    with open(test_file, "w") as f: f.write("line1\nline2\nline3\nline4\nline5")
    assert get_source_snippet(str(test_file), 2, 4) == "line2\nline3\nline4\n"
    assert get_source_snippet(str(test_file), 2, None, max_lines_override=2) == "line2\nline3\n"

def test_context_retriever_initialization(retriever_fixture_data):
    kg, all_modules, codebase_root = retriever_fixture_data
    retriever = ContextRetriever(kg, all_modules, codebase_root)
    assert retriever.kg is not None and retriever.all_modules is not None

def test_context_retriever_format_signatures(retriever_fixture_data):
    kg, all_modules, codebase_root = retriever_fixture_data
    retriever = ContextRetriever(kg, all_modules, codebase_root)
    utils_helpers_abs_path = os.path.join(codebase_root, "utils", "helpers.py")

    func_node_id = get_function_node_id(utils_helpers_abs_path, "format_user")
    assert "def format_user(name: str, age: int) -> str:" in retriever._format_function_signature(func_node_id)

    class_node_id = get_class_node_id(utils_helpers_abs_path, "StringHelper")
    assert "class StringHelper:" in retriever._format_class_signature(class_node_id)

def test_context_retriever_get_formatted_definitions(retriever_fixture_data):
    kg, all_modules, codebase_root = retriever_fixture_data
    retriever = ContextRetriever(kg, all_modules, codebase_root)
    utils_helpers_abs_path = os.path.join(codebase_root_path, "utils", "helpers.py") # Use codebase_root_path from outer scope

    func_node_id = get_function_node_id(utils_helpers_abs_path, "format_user")
    definition = retriever._get_formatted_definition(func_node_id)
    assert "def format_user(name: str, age: int) -> str:" in definition
    assert "Formats user info." in definition
    assert "sh = StringHelper()" in definition

    class_node_id = get_class_node_id(utils_helpers_abs_path, "StringHelper")
    definition = retriever._get_formatted_definition(class_node_id)
    assert "class StringHelper:" in definition
    assert "Helper for strings." in definition
    assert "def capitalize_first(self, text: str) -> str:" in definition

def test_context_retriever_get_context_for_prompt_target_function(retriever_fixture_data):
    kg, all_modules, codebase_root = retriever_fixture_data
    retriever = ContextRetriever(kg, all_modules, codebase_root)

    current_file_rel = os.path.join("utils", "helpers.py")
    prompt = "Explain the function format_user in utils/helpers.py"

    context = retriever.get_context_for_prompt(prompt, current_filepath_rel=current_file_rel)
    assert "Context: function `format_user` from `utils/helpers.py`" in context
    assert "def format_user(name: str, age: int) -> str:" in context
    assert "Functions/Methods Called By It:" in context
    # Ensure the mock call to StringHelper.capitalize_first is part of the context
    assert "StringHelper.capitalize_first" in context # Based on manual resolution in fixture

def test_context_retriever_domain_keyword_test(retriever_fixture_data):
    kg, all_modules, codebase_root = retriever_fixture_data
    retriever = ContextRetriever(kg, all_modules, codebase_root)
    current_file_rel = os.path.join("utils", "helpers.py")
    prompt = "Write a pytest test for format_user function."
    context = retriever.get_context_for_prompt(prompt, current_filepath_rel=current_file_rel)
    assert "Context: function `format_user` from `utils/helpers.py`" in context
    # With no callers for format_user, the "test" domain heuristic might not add much different context here.
    # This test confirms the domain detection runs and doesn't break basic context retrieval.
    assert "Functions/Methods Called By It:" in context # Default related context (callees)
    assert "StringHelper.capitalize_first" in context
