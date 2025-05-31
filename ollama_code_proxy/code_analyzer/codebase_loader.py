import os
from typing import List, Dict, Tuple, Optional, Callable, Any # Added Any
import networkx as nx

from .parser import PythonParser
from .models import ModuleInfo
from .reference_resolver import ReferenceResolver
from .knowledge_graph import KnowledgeGraph

class CodebaseLoader:
    def __init__(
        self,
        codebase_root: str,
        parser: Optional[PythonParser] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        self.codebase_root = os.path.abspath(codebase_root)
        self.parser = parser if parser else PythonParser()
        self.progress_callback = progress_callback

        self.loaded_modules: Dict[str, ModuleInfo] = {}
        self.parsing_errors: List[Dict[str, Any]] = []
        self.knowledge_graph: Optional[nx.DiGraph] = None

    def _is_excluded(self, current_item_abs_path: str, item_name: str, is_dir: bool,
                     effective_exclude_dirs: List[str], effective_exclude_files: List[str]) -> bool:
        if is_dir:
            if item_name in effective_exclude_dirs or item_name.startswith('.'): # Exclude hidden directories
                return True
            try:
                relative_path = os.path.relpath(current_item_abs_path, self.codebase_root)
                if relative_path == ".": return False
                for part in relative_path.split(os.sep):
                    if part in effective_exclude_dirs or part.startswith('.'): # Check parts for exclusion
                        return True
            except ValueError:
                return True
        else: # File
            if item_name in effective_exclude_files or item_name.startswith('.'): # Exclude hidden files
                return True
        return False

    def load_analyze_and_build_graph(
        self,
        exclude_dirs: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None
    ) -> None:
        if not os.path.isdir(self.codebase_root):
            raise ValueError(f"Codebase path {self.codebase_root} is not a valid directory.")

        default_exclude_dirs = [
            'venv', '.venv', 'env', '.env', '__pycache__', '.git', '.hg', '.svn',
            'node_modules', 'target', 'build', 'dist', 'docs', 'site-packages',
            'lib', 'lib64', 'test', 'tests', 'tmp', 'temp',
            '.pytest_cache', '.mypy_cache', '*.egg-info'
        ]
        effective_exclude_dirs = exclude_dirs if exclude_dirs is not None else default_exclude_dirs

        default_exclude_files = ['.DS_Store', '*.pyc', '*.pyo', '*.pyd', '*.so', 'setup.py', 'conftest.py']
        effective_exclude_files = exclude_files if exclude_files is not None else default_exclude_files

        self.loaded_modules = {}
        self.parsing_errors = []
        self.knowledge_graph = None

        filepaths_to_parse: List[str] = []
        for root, dirs, files in os.walk(self.codebase_root, topdown=True):
            dirs[:] = [d for d in dirs if not self._is_excluded(os.path.abspath(os.path.join(root, d)), d, True, effective_exclude_dirs, effective_exclude_files)]

            for file_name in files:
                abs_filepath = os.path.abspath(os.path.join(root, file_name))
                if file_name.endswith(".py") and not self._is_excluded(abs_filepath, file_name, False, effective_exclude_dirs, effective_exclude_files):
                    filepaths_to_parse.append(abs_filepath)

        total_files = len(filepaths_to_parse)
        current_phase_msg = "Parsing files"
        if self.progress_callback: self.progress_callback(0, total_files, current_phase_msg)

        for i, abs_filepath in enumerate(filepaths_to_parse):
            if self.progress_callback:
                self.progress_callback(i + 1, total_files, f"{current_phase_msg}")

            raw_parsed_data = self.parser.parse(abs_filepath)

            parser_errors = raw_parsed_data.get("parse_errors", [])
            if parser_errors:
                for err_detail in parser_errors:
                     self.parsing_errors.append({
                        "filepath": abs_filepath,
                        "error": err_detail.get("message", "Unknown parsing error from parser"),
                        "lineno": err_detail.get("lineno"), "offset": err_detail.get("offset"),
                        "text": err_detail.get("text", "")
                    })

            if any(err.get("type") == "FileAccessError" for err in parser_errors):
                continue

            try:
                # Ensure 'filepath' in raw_parsed_data is relative for ModuleInfo object
                # The parser returns absolute path in its "filepath" field.
                parser_fp = raw_parsed_data.get("filepath")
                if parser_fp and os.path.isabs(parser_fp):
                     raw_parsed_data["filepath"] = os.path.relpath(parser_fp, self.codebase_root)
                elif not parser_fp : # If parser somehow didn't set it
                     raw_parsed_data["filepath"] = os.path.relpath(abs_filepath, self.codebase_root)


                module_info = ModuleInfo(**raw_parsed_data)
                self.loaded_modules[abs_filepath] = module_info
            except Exception as e:
                self.parsing_errors.append({
                    "filepath": abs_filepath, "error": f"Failed to create ModuleInfo from parsed data: {str(e)}",
                    "lineno": None, "offset": None, "text": ""
                })

        if self.progress_callback: self.progress_callback(total_files, total_files, "Parsing complete. Resolving references...")

        if self.loaded_modules:
            resolver = ReferenceResolver(all_modules=self.loaded_modules, codebase_root=self.codebase_root)
            resolver.resolve_all_references()
            if self.progress_callback: self.progress_callback(total_files, total_files, "Reference resolution complete. Building graph...")

            kg_builder = KnowledgeGraph(all_modules=self.loaded_modules, codebase_root=self.codebase_root)
            self.knowledge_graph = kg_builder.get_graph()
            if self.progress_callback: self.progress_callback(total_files, total_files, "Knowledge graph built.")

        if self.progress_callback: self.progress_callback(total_files, total_files, "Analysis complete.")

    def get_module(self, filepath: str) -> Optional[ModuleInfo]:
        return self.loaded_modules.get(os.path.abspath(filepath))

    def get_all_modules(self) -> Dict[str, ModuleInfo]:
        return self.loaded_modules

    def get_parsing_errors(self) -> List[Dict[str, Any]]:
        return self.parsing_errors

    def get_knowledge_graph(self) -> Optional[nx.DiGraph]:
        return self.knowledge_graph

if __name__ == '__main__':
    import sys
    import shutil
    project_root_for_example = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root_for_example not in sys.path:
        sys.path.insert(0, project_root_for_example)

    from ollama_code_proxy.code_analyzer.parser import PythonParser
    from ollama_code_proxy.code_analyzer.models import ModuleInfo
    from ollama_code_proxy.code_analyzer.reference_resolver import ReferenceResolver
    from ollama_code_proxy.code_analyzer.knowledge_graph import KnowledgeGraph


    print("Running CodebaseLoader __main__ example with KG integration...")

    test_root = os.path.abspath("temp_loader_kg_test_project")
    if os.path.exists(test_root): shutil.rmtree(test_root)
    module_a_path = os.path.join(test_root, "module_a.py")
    pkg_dir = os.path.join(test_root, "pkg")
    module_b_path = os.path.join(pkg_dir, "module_b.py")
    pkg_init_path = os.path.join(pkg_dir, "__init__.py")

    os.makedirs(pkg_dir, exist_ok=True)

    with open(module_a_path, "w", encoding="utf-8") as f:
        f.write("""
from pkg.module_b import func_b, ClassB
from .pkg import module_b # This import might be problematic for simple resolver if module_a is top-level
import os

class MyClassA: pass

def func_a1():
    x = ClassB()
    y = MyClassA()
    return func_b(os.sep)

def func_a2():
    return module_b.func_b("test")
""")
    with open(module_b_path, "w", encoding="utf-8") as f:
        f.write("""
from ..module_a import MyClassA

class ClassB:
    def method_b(self):
        a = MyClassA()
        return "method_b_called"

def func_b(param):
    c = ClassB()
    return c.method_b() + str(param)
""")
    with open(pkg_init_path, "w", encoding="utf-8") as f: f.write("from .module_b import ClassB, func_b")

    def simple_progress(cur, total, phase):
        print(f"Progress: {phase} - {cur}/{total}")

    loader = CodebaseLoader(codebase_root=test_root, progress_callback=simple_progress)
    try:
        print(f"\nLoading codebase at: {test_root}")
        loader.load_analyze_and_build_graph(
            exclude_dirs=['.git', '.venv', 'test_excluded_dir'],
            exclude_files=['excluded_file.py']
        )

        print(f"\n--- Loaded {len(loader.get_all_modules())} modules ---")
        for abs_path, mod_info in loader.get_all_modules().items():
            print(f"\nModule: {mod_info.module_name} (Rel Path: {mod_info.filepath})")
            if mod_info.imports:
                for imp in mod_info.imports:
                    print(f"  Import: name='{imp.name}', module='{imp.module}', level={imp.level}, Resolved: {imp.resolved_filepath}")

            for func in mod_info.functions:
                print(f"  Function: {func.name}")
                for call in func.function_calls:
                    print(f"    Call: '{call.target_name}', RP: {call.resolved_target_filepath}, RN: {call.resolved_target_name}")
                for inst in func.instance_creations:
                    print(f"    Instance: '{inst.class_name}', RP: {inst.resolved_target_filepath}")

            for cls in mod_info.classes:
                 print(f"  Class: {cls.name}")
                 if cls.resolved_bases:
                     for base in cls.resolved_bases:
                         print(f"    Base: '{base['name']}', RP: {base['filepath']}")
                 for meth in cls.methods:
                    print(f"    Method: {meth.name}")
                    for call in meth.function_calls:
                        print(f"      Call: '{call.target_name}', RP: {call.resolved_target_filepath}, RN: {call.resolved_target_name}")
                    for inst in meth.instance_creations:
                        print(f"      Instance: '{inst.class_name}', RP: {inst.resolved_target_filepath}")

        kg = loader.get_knowledge_graph()
        if kg:
            print(f"\nKnowledge Graph built: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")
        else:
            print("\nKnowledge Graph not built (no modules loaded or error).")

        if loader.get_parsing_errors():
            print("\n--- Parsing Errors Reported by Loader ---")
            for err in loader.get_parsing_errors():
                print(f"  File: {err['filepath']}, Line: {err.get('lineno')}, Error: {err['error']}")

    except Exception as e:
        print(f"Error during example: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_root):
            shutil.rmtree(test_root)
        print(f"\nCleaned up: {test_root}")

    print("\nCodebaseLoader example finished.")
