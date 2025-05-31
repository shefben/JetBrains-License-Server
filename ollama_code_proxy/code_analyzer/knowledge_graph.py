import networkx as nx
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import os

from .models import ModuleInfo, ClassInfo, FunctionInfo, MethodInfo, CallInfo, InstanceCreationInfo, ImportInfo

# --- Node ID Generation Functions ---
def get_module_node_id(abs_filepath: str) -> str:
    return abs_filepath

def get_class_node_id(module_abs_filepath: str, class_name: str) -> str:
    return f"{module_abs_filepath}::{class_name}"

def get_function_node_id(module_abs_filepath: str, function_name: str) -> str:
    return f"{module_abs_filepath}::{function_name}"

def get_method_node_id(module_abs_filepath: str, class_name: str, method_name: str) -> str:
    return f"{module_abs_filepath}::{class_name}::{method_name}"


class KnowledgeGraph:
    def __init__(self, all_modules: Dict[str, ModuleInfo], codebase_root: str):
        self.all_modules: Dict[str, ModuleInfo] = all_modules
        self.codebase_root: str = os.path.abspath(codebase_root)
        self.graph: nx.DiGraph = nx.DiGraph()
        if all_modules: # Only build if there's data
            self._build_graph()

    def _add_module_node(self, module_abs_filepath: str, module_info: ModuleInfo):
        node_id = get_module_node_id(module_abs_filepath)
        self.graph.add_node(
            node_id,
            type="module",
            name=module_info.module_name,
            filepath_abs=module_abs_filepath,
            filepath_rel=module_info.filepath,
            docstring=module_info.docstring,
            language=module_info.language
        )

    def _add_class_node(self, module_abs_filepath: str, class_info: ClassInfo):
        node_id = get_class_node_id(module_abs_filepath, class_info.name)
        self.graph.add_node(
            node_id,
            type="class",
            name=class_info.name,
            filepath_abs=module_abs_filepath,
            filepath_rel=os.path.relpath(module_abs_filepath, self.codebase_root),
            lineno=class_info.lineno,
            end_lineno=class_info.end_lineno,
            docstring=class_info.docstring,
            decorators=[d.name for d in class_info.decorators]
        )
        module_node_id = get_module_node_id(module_abs_filepath)
        if self.graph.has_node(module_node_id): # Ensure module node exists
            self.graph.add_edge(module_node_id, node_id, type="defines_class")

    def _add_function_node(self, module_abs_filepath: str, func_info: FunctionInfo):
        node_id = get_function_node_id(module_abs_filepath, func_info.name)
        self.graph.add_node(
            node_id,
            type="function",
            name=func_info.name,
            filepath_abs=module_abs_filepath,
            filepath_rel=os.path.relpath(module_abs_filepath, self.codebase_root),
            lineno=func_info.lineno,
            end_lineno=func_info.end_lineno,
            docstring=func_info.docstring,
            is_async=func_info.is_async,
            decorators=[d.name for d in func_info.decorators]
        )
        module_node_id = get_module_node_id(module_abs_filepath)
        if self.graph.has_node(module_node_id):
            self.graph.add_edge(module_node_id, node_id, type="defines_function")

    def _add_method_node(self, module_abs_filepath: str, class_name: str, method_info: MethodInfo):
        node_id = get_method_node_id(module_abs_filepath, class_name, method_info.name)
        self.graph.add_node(
            node_id,
            type="method",
            name=method_info.name,
            class_name=class_name,
            filepath_abs=module_abs_filepath,
            filepath_rel=os.path.relpath(module_abs_filepath, self.codebase_root),
            lineno=method_info.lineno,
            end_lineno=method_info.end_lineno,
            docstring=method_info.docstring,
            is_async=method_info.is_async,
            is_static=method_info.is_static,
            is_classmethod=method_info.is_classmethod,
            decorators=[d.name for d in method_info.decorators]
        )
        class_node_id = get_class_node_id(module_abs_filepath, class_name)
        if self.graph.has_node(class_node_id):
             self.graph.add_edge(class_node_id, node_id, type="defines_method")

    def _build_graph(self):
        for module_abs_filepath, module_info in self.all_modules.items():
            self._add_module_node(module_abs_filepath, module_info)
            for func_info in module_info.functions:
                self._add_function_node(module_abs_filepath, func_info)
            for class_info in module_info.classes:
                self._add_class_node(module_abs_filepath, class_info)
                for method_info in class_info.methods:
                    self._add_method_node(module_abs_filepath, class_info.name, method_info)

        for module_abs_filepath, module_info in self.all_modules.items():
            module_node_id = get_module_node_id(module_abs_filepath)
            if not self.graph.has_node(module_node_id): continue # Should exist, but defensive check

            for imp_info in module_info.imports:
                if imp_info.resolved_filepath:
                    resolved_abs_imported_module_path = os.path.normpath(os.path.join(self.codebase_root, imp_info.resolved_filepath))
                    imported_module_node_id = get_module_node_id(resolved_abs_imported_module_path)
                    if self.graph.has_node(imported_module_node_id):
                         self.graph.add_edge(module_node_id, imported_module_node_id, type="imports",
                                             imported_name=imp_info.name, alias=imp_info.asname)

            elements_with_calls: List[Union[FunctionInfo, MethodInfo]] = [] # type: ignore
            elements_with_calls.extend(module_info.functions)
            for cls in module_info.classes: elements_with_calls.extend(cls.methods)

            for element in elements_with_calls:
                parent_class_name = None
                if isinstance(element, MethodInfo):
                    for c_info in module_info.classes:
                        if element in c_info.methods:
                            parent_class_name = c_info.name; break
                    if not parent_class_name: continue
                    caller_node_id = get_method_node_id(module_abs_filepath, parent_class_name, element.name)
                else:
                    caller_node_id = get_function_node_id(module_abs_filepath, element.name)

                if not self.graph.has_node(caller_node_id): continue

                for call in element.function_calls:
                    if call.resolved_target_filepath and call.resolved_target_name:
                        resolved_abs_callee_module_path = os.path.normpath(os.path.join(self.codebase_root, call.resolved_target_filepath))
                        callee_node_id: Optional[str] = None
                        if call.is_method_call and "." in call.resolved_target_name:
                             target_class_name, target_method_name = call.resolved_target_name.rsplit('.',1)
                             callee_node_id = get_method_node_id(resolved_abs_callee_module_path, target_class_name, target_method_name)
                        elif not call.is_method_call :
                             callee_node_id = get_function_node_id(resolved_abs_callee_module_path, call.resolved_target_name)

                        if callee_node_id and self.graph.has_node(callee_node_id):
                             self.graph.add_edge(caller_node_id, callee_node_id, type="calls", lineno=call.lineno)

                for inst in element.instance_creations:
                    if inst.resolved_target_filepath:
                        resolved_abs_class_module_path = os.path.normpath(os.path.join(self.codebase_root, inst.resolved_target_filepath))
                        created_class_node_id = get_class_node_id(resolved_abs_class_module_path, inst.class_name)
                        if self.graph.has_node(created_class_node_id):
                             self.graph.add_edge(caller_node_id, created_class_node_id, type="creates_instance", lineno=inst.lineno)

            for class_info_item in module_info.classes:
                current_class_node_id = get_class_node_id(module_abs_filepath, class_info_item.name)
                if not self.graph.has_node(current_class_node_id): continue
                for base_info in class_info_item.resolved_bases:
                    if base_info["filepath"] and base_info["name"]:
                        resolved_abs_base_module_path = os.path.normpath(os.path.join(self.codebase_root, base_info["filepath"]))
                        base_class_node_id = get_class_node_id(resolved_abs_base_module_path, base_info["name"])
                        if self.graph.has_node(base_class_node_id):
                            self.graph.add_edge(current_class_node_id, base_class_node_id, type="inherits_from")

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def get_node_attributes(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return None

    def find_nodes(self, name: Optional[str] = None, type: Optional[str] = None, filepath_rel: Optional[str] = None) -> List[str]:
        results: List[str] = []
        for node_id, attrs in self.graph.nodes(data=True):
            match_name = (name is None) or (attrs.get('name') == name)
            match_type = (type is None) or (attrs.get('type') == type)
            match_filepath = (filepath_rel is None) or (attrs.get('filepath_rel') == filepath_rel)
            if match_name and match_type and match_filepath:
                results.append(node_id)
        return results

    def get_callers(self, target_node_id: str) -> List[str]:
        if not self.graph.has_node(target_node_id): return []
        return [u for u, v, data in self.graph.in_edges(target_node_id, data=True) if data.get('type') == 'calls']

    def get_callees(self, source_node_id: str) -> List[str]:
        if not self.graph.has_node(source_node_id): return []
        return [v for u, v, data in self.graph.out_edges(source_node_id, data=True) if data.get('type') == 'calls']

    def get_inheritance_parents(self, class_node_id: str) -> List[str]:
        if not self.graph.has_node(class_node_id): return []
        return [v for u, v, data in self.graph.out_edges(class_node_id, data=True) if data.get('type') == 'inherits_from']

    def get_inheritance_children(self, class_node_id: str) -> List[str]:
        if not self.graph.has_node(class_node_id): return []
        return [u for u, v, data in self.graph.in_edges(class_node_id, data=True) if data.get('type') == 'inherits_from']

    def get_all_ancestors(self, class_node_id: str) -> Set[str]:
        if not self.graph.has_node(class_node_id): return set()
        return nx.ancestors(self.graph, class_node_id) # Assumes edges child -> parent

    def get_all_descendants(self, class_node_id: str) -> Set[str]:
        if not self.graph.has_node(class_node_id): return set()
        descendants = set()
        q = list(self.get_inheritance_children(class_node_id))
        visited = set(q)
        descendants.update(q)
        head = 0
        while head < len(q):
            curr = q[head]; head += 1
            for child in self.get_inheritance_children(curr):
                if child not in visited:
                    visited.add(child); descendants.add(child); q.append(child) # Corrected bug here
        return descendants

    def find_definitions(self, symbol_name: str, module_filepath_rel: Optional[str] = None) -> List[str]:
        results: List[str] = []
        for node_id, attrs in self.graph.nodes(data=True):
            is_target_module = (module_filepath_rel is None) or (attrs.get('filepath_rel') == module_filepath_rel)
            if not is_target_module: continue
            if attrs.get('name') == symbol_name and attrs.get('type') in ['function', 'class', 'method']:
                results.append(node_id)
        return results

    def find_references_to_node(self, target_node_id: str) -> Dict[str, List[Any]]:
        if not self.graph.has_node(target_node_id): return {}
        refs: Dict[str, List[Any]] = {
            "called_by": [], "instances_created_in": [], "imported_by_modules": [],
            "subclasses": [], "superclasses": []
        }
        target_attrs = self.graph.nodes[target_node_id]
        target_type = target_attrs.get("type")

        if target_type in ["function", "method"]:
            refs["called_by"] = self.get_callers(target_node_id)
        if target_type == "class":
            for u, v, data in self.graph.in_edges(target_node_id, data=True):
                if data.get("type") == "creates_instance": refs["instances_created_in"].append(u)
                elif data.get("type") == "inherits_from": refs["subclasses"].append(u)
            refs["superclasses"] = self.get_inheritance_parents(target_node_id)
        if target_type == "module":
            for u, v, data in self.graph.in_edges(target_node_id, data=True):
                if data.get("type") == "imports":
                    refs["imported_by_modules"].append({"importer_module_id": u, "details": data})
        return {k: v for k, v in refs.items() if v}

if __name__ == '__main__':
    import sys
    import shutil
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
    if project_root_dir not in sys.path: sys.path.insert(0, project_root_dir)
    from ollama_code_proxy.code_analyzer.codebase_loader import CodebaseLoader # type: ignore

    print("KnowledgeGraph class with query methods defined.")
    # Example usage:
    # test_project_path = os.path.join(current_script_dir, "temp_kg_test_project_queries")
    # ... (setup dummy project as in CodebaseLoader's main) ...
    # loader = CodebaseLoader(test_project_path)
    # loader.load_analyze_and_build_graph()
    # kg_instance = loader.get_knowledge_graph()
    # if kg_instance:
    #     print(f"Graph: {kg_instance.number_of_nodes()} nodes, {kg_instance.number_of_edges()} edges.")
    # if os.path.exists(test_project_path): shutil.rmtree(test_project_path)
