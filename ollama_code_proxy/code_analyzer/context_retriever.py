import os
import re
from typing import Dict, List, Optional, Any, Set, Tuple, Union

from .knowledge_graph import KnowledgeGraph
from .models import ModuleInfo, FunctionInfo, MethodInfo, ClassInfo, DecoratorInfo, ArgumentInfo

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def get_source_snippet(
    filepath_abs: str,
    start_lineno: int,
    end_lineno: Optional[int] = None,
    max_lines_override: Optional[int] = None
) -> Optional[str]:
    try:
        if not os.path.isabs(filepath_abs): return None
        with open(filepath_abs, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        if not (1 <= start_lineno <= len(lines)): return None

        effective_end_lineno = end_lineno
        default_snippet_lines = 20

        if max_lines_override is not None:
            effective_end_lineno = start_lineno + max_lines_override - 1
        elif effective_end_lineno is None:
             effective_end_lineno = start_lineno + default_snippet_lines - 1

        effective_end_lineno = min(effective_end_lineno, len(lines))
        if effective_end_lineno < start_lineno : effective_end_lineno = start_lineno

        snippet_lines = lines[start_lineno-1 : effective_end_lineno]
        return "".join(snippet_lines)
    except FileNotFoundError: return None
    except Exception: return None

class ContextRetriever:
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "database": ["database", "db", "sql", "sqlalchemy", "model", "query", "orm", "table", "schema", "migrate", "migration"],
        "api": ["api", "route", "endpoint", "fastapi", "flask", "django", "http", "request", "response", "decorator", "json", "get", "post", "put", "delete", "url"],
        "test": ["test", "pytest", "unittest", "assert", "fixture", "mock", "patch", "should", "expect", "assertEqual"],
        "ui": ["ui", "frontend", "react", "vue", "angular", "javascript", "css", "html", "component", "view", "template", "dom", "render"],
        "data_processing": ["pandas", "numpy", "dataframe", "series", "array", "csv", "excel", "etl", "pipeline", "transform", "load", "spark", "hadoop"],
        "concurrency": ["async", "await", "thread", "process", "concurrent", "parallel", "lock", "semaphore", "queue", "actor", "coroutine"],
        "file_io": ["file", "path", "open", "read", "write", "io", "os.path", "shutil", "glob"],
        "security": ["security", "auth", "authentication", "authorization", "token", "jwt", "oauth", "encrypt", "decrypt", "ssl", "tls", "vulnerability"],
        "logging": ["log", "logging", "logger", "loguru", "sentry", "debug", "info", "warning", "error", "critical"]
    }

    def __init__(self, kg: KnowledgeGraph, all_modules: Dict[str, ModuleInfo], codebase_root: str):
        self.kg = kg
        self.all_modules = all_modules
        self.codebase_root = os.path.abspath(codebase_root)

    def _format_function_signature(self, func_node_id: str) -> Optional[str]:
        attrs = self.kg.get_node_attributes(func_node_id)
        if not attrs or attrs.get("type") not in ["function", "method"]: return None

        module_abs_filepath = attrs.get("filepath_abs")
        if not module_abs_filepath: return f"def {attrs['name']}(...): # Filepath missing"

        module_info = self.all_modules.get(module_abs_filepath)
        if not module_info: return f"def {attrs['name']}(...): # ModuleInfo not found"

        target_func_info: Optional[Union[FunctionInfo, MethodInfo]] = None # type: ignore
        func_name = attrs["name"]
        if attrs.get("type") == "function":
            target_func_info = next((f for f in module_info.functions if f.name == func_name), None)
        elif attrs.get("type") == "method" and attrs.get("class_name"):
            class_name = attrs["class_name"]
            parent_class_info = next((c for c in module_info.classes if c.name == class_name), None)
            if parent_class_info:
                target_func_info = next((m for m in parent_class_info.methods if m.name == func_name), None)

        if not target_func_info: return f"def {func_name}(...): # Signature details not found"

        args_str_parts = [f"{arg.name}{f': {arg.annotation}' if arg.annotation else ''}{f' = {arg.default_value}' if arg.default_value is not None else ''}" for arg in target_func_info.args]
        args_str = ", ".join(args_str_parts)
        return_annot = f" -> {target_func_info.returns}" if target_func_info.returns else ""
        async_prefix = "async " if target_func_info.is_async else ""

        decorator_strs = [f"@{dec.name}" for dec in target_func_info.decorators]
        # Add @staticmethod or @classmethod if they are true on MethodInfo and not already in decorators from parser
        if isinstance(target_func_info, MethodInfo):
            if target_func_info.is_classmethod and not any("@classmethod" in d for d in decorator_strs):
                decorator_strs.insert(0, "@classmethod")
            if target_func_info.is_static and not any("@staticmethod" in d for d in decorator_strs):
                decorator_strs.insert(0, "@staticmethod")

        full_decorator_str = "\n".join(decorator_strs) + "\n" if decorator_strs else ""
        return f"{full_decorator_str}{async_prefix}def {func_name}({args_str}){return_annot}:"

    def _format_class_signature(self, class_node_id: str) -> Optional[str]:
        attrs = self.kg.get_node_attributes(class_node_id)
        if not attrs or attrs.get("type") != "class": return None
        module_abs_filepath = attrs.get("filepath_abs")
        if not module_abs_filepath: return f"class {attrs['name']}: # Filepath missing"
        module_info = self.all_modules.get(module_abs_filepath)
        if not module_info: return f"class {attrs['name']}: # ModuleInfo not found"
        class_info_model = next((c for c in module_info.classes if c.name == attrs["name"]), None)
        if not class_info_model: return f"class {attrs['name']}: # ClassInfo details not found"

        bases_str = f"({', '.join(class_info_model.bases)})" if class_info_model.bases else ""
        decorator_strs = [f"@{dec.name}" for dec in class_info_model.decorators]
        full_decorator_str = "\n".join(decorator_strs) + "\n" if decorator_strs else ""
        return f"{full_decorator_str}class {attrs['name']}{bases_str}:"

    def _get_formatted_definition(
        self, node_id: str, is_primary_target: bool = True,
        max_body_lines: int = 10, max_methods_in_class_snippet: int = 5
    ) -> Optional[str]:
        attrs = self.kg.get_node_attributes(node_id)
        if not attrs: return None
        output_parts: List[str] = []
        signature: Optional[str] = None
        node_type, node_name, node_abs_filepath = attrs.get("type"), attrs.get("name"), attrs.get("filepath_abs")
        if not all([node_type, node_name, node_abs_filepath]): return f"# Invalid node data for {node_id}"

        node_rel_filepath = os.path.relpath(node_abs_filepath, self.codebase_root)
        header_prefix = "Primary" if is_primary_target else "Related"
        header = f"### {header_prefix} Context: {node_type} `{node_name}` from `{node_rel_filepath}`:\n```python"
        output_parts.append(header)

        if node_type in ["function", "method"]: signature = self._format_function_signature(node_id)
        elif node_type == "class": signature = self._format_class_signature(node_id)

        if not signature: output_parts.extend([f"# Could not format signature for {node_name}", "```"]); return "\n".join(output_parts)
        output_parts.append(signature)

        docstring = attrs.get("docstring")
        if docstring:
            indented_docstring = "\n".join([f"    {line}" for line in docstring.strip().splitlines()])
            output_parts.append(f'    """{indented_docstring}"""' if '"""' not in docstring and "'''" not in docstring else f"    {docstring.strip()}")

        body_lines_to_show = max_body_lines

        if node_type == "class":
            module_info = self.all_modules.get(node_abs_filepath)
            class_model = next((c for c in module_info.classes if c.name == node_name), None) if module_info else None
            if class_model:
                method_count = 0
                for method_model in class_model.methods:
                    if method_count >= max_methods_in_class_snippet and not is_primary_target: output_parts.append("    ..."); break
                    method_node_id_val = get_method_node_id(node_abs_filepath, node_name, method_model.name)
                    # Check if method node actually exists in graph (it should if graph built correctly)
                    if self.kg.graph.has_node(method_node_id_val):
                        method_sig = self._format_function_signature(method_node_id_val)
                        if method_sig:
                            indented_sig = "\n".join([f"    {line}" for line in method_sig.splitlines()])
                            output_parts.append(f"\n{indented_sig}")
                            if method_model.docstring: output_parts.append(f"        \"\"\"{method_model.docstring.strip()}\"\"\"")
                            output_parts.append("        ...")
                            method_count +=1
                if not class_model.methods and is_primary_target : output_parts.append("    pass # No methods in snippet")

        elif node_type in ["function", "method"] and body_lines_to_show > 0:
            start_line, end_line = attrs.get("lineno", 0), attrs.get("end_lineno")
            # Approximate body start: line after signature and docstring
            body_start_line = start_line + signature.count('\n') + 1
            if docstring: body_start_line += docstring.count('\n') + 1

            body_snippet = get_source_snippet(node_abs_filepath, body_start_line, max_lines_override=body_lines_to_show)
            if body_snippet:
                output_parts.extend([f"    {line}" for line in body_snippet.strip().splitlines() if line.strip()])
                num_snippet_lines = body_snippet.count('\n') +1 # Number of lines in the snippet
                # If end_line is known and the snippet doesn't reach it, add "..."
                if end_line and (body_start_line + num_snippet_lines -1) < end_line : output_parts.append("    ...")
            elif is_primary_target: output_parts.append("    pass # Body not shown or empty")

        output_parts.append("```")
        return "\n".join(output_parts)

    def _extract_potential_symbols_from_prompt(self, user_prompt: str) -> Tuple[Set[str], Set[str]]:
        symbols, filepaths = set(), set()
        # Regex for Python identifiers and dot-separated names
        for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)\b', user_prompt):
            token = match.group(1)
            # Exclude common keywords and very short tokens unless they are part of a dotted path
            if '.' in token and token.lower().endswith('.py'): filepaths.add(token.replace('\\', '/'))
            elif token.lower() not in {"def", "class", "import", "from", "return", "yield", "async", "await", "self", "cls", "pass", "is", "in", "and", "or", "not", "true", "false", "none"}:
                if len(token) > 2 or '.' in token: # Allow short names if part of dotted path
                    symbols.add(token)
        # Regex for file paths (more general)
        for match in re.finditer(r'\b([\w./\\_~-]+(\.py|\.md|\.txt|\.json|\.yaml|\.yml))\b', user_prompt):
            filepaths.add(match.group(1).replace('\\', '/'))
        return symbols, filepaths

    def _detect_domain_keywords(self, user_prompt: str) -> Set[str]:
        detected_domains = set()
        lower_prompt = user_prompt.lower()
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(f" {keyword} " in lower_prompt or lower_prompt.startswith(keyword + " ") or lower_prompt.endswith(" " + keyword) or lower_prompt == keyword for keyword in keywords): # Added exact match for single word prompt
                detected_domains.add(domain)
        return detected_domains

    def get_context_for_prompt(
        self, user_prompt: str, current_filepath_rel: Optional[str] = None, max_context_tokens: int = 1500
    ) -> str:
        context_items_with_scores: List[Tuple[float, str]] = []
        current_tokens = 0
        processed_node_ids: Set[str] = set()

        prompt_symbols, prompt_filepaths = self._extract_potential_symbols_from_prompt(user_prompt)
        detected_domains = self._detect_domain_keywords(user_prompt)

        primary_target_node_ids: Set[str] = set()
        current_file_abs_path: Optional[str] = os.path.normpath(os.path.join(self.codebase_root, current_filepath_rel)) if current_filepath_rel else None

        for fp_str in prompt_filepaths:
            path_to_check = os.path.normpath(fp_str)
            if not os.path.isabs(path_to_check):
                base_dir = os.path.dirname(current_file_abs_path) if current_file_abs_path else self.codebase_root
                path_to_check = os.path.normpath(os.path.join(base_dir, fp_str))

            if os.path.exists(path_to_check) and path_to_check.startswith(self.codebase_root) and self.kg.graph.has_node(path_to_check):
                primary_target_node_ids.add(path_to_check)

        for sym_name in prompt_symbols:
            cand_nodes = []
            if current_file_abs_path: # Prioritize current file if context is available
                cand_nodes.extend(self.kg.find_definitions(sym_name, module_filepath_rel=os.path.relpath(current_file_abs_path, self.codebase_root)))
            if not cand_nodes: # Fallback to global search
                cand_nodes.extend(self.kg.find_definitions(sym_name))
            if cand_nodes: primary_target_node_ids.update(cand_nodes[:2])

        unique_primary_targets = list(set(primary_target_node_ids)) # Deduplicate

        for node_id in unique_primary_targets:
            if node_id in processed_node_ids or current_tokens >= max_context_tokens: break
            definition_str = self._get_formatted_definition(node_id, is_primary_target=True, max_body_lines=15, max_methods_in_class_snippet=5)
            if definition_str:
                item_tokens = estimate_tokens(definition_str)
                if current_tokens + item_tokens <= max_context_tokens:
                    context_items_with_scores.append((1.0, definition_str)); current_tokens += item_tokens; processed_node_ids.add(node_id)

        # Heuristic expansion
        for node_id in list(processed_node_ids):
            if current_tokens >= max_context_tokens: break
            attrs = self.kg.get_node_attributes(node_id)
            if not attrs : continue

            related_to_fetch: List[Tuple[str, float]] = []
            node_type = attrs.get("type")

            if node_type in ["function", "method"]:
                if "test" in detected_domains: related_to_fetch.extend([(c, 0.85) for c in self.kg.get_callers(node_id)[:2]]) # Show 2 callers for tests
                else: related_to_fetch.extend([(c, 0.8) for c in self.kg.get_callees(node_id)[:2]]) # Show 2 callees otherwise

            if node_type == "class":
                 related_to_fetch.extend([(p, 0.75) for p in self.kg.get_inheritance_parents(node_id)[:1]]) # Show 1 direct parent
                 if "database" in detected_domains or "api" in detected_domains : # Show more methods for these domains
                     # This would ideally get method nodes, not just re-format current class.
                     # For now, this example doesn't add extra methods beyond what _get_formatted_definition does.
                     pass


            for rel_node_id, score in related_to_fetch:
                if rel_node_id in processed_node_ids or current_tokens >= max_context_tokens: continue
                rel_def_str = self._get_formatted_definition(rel_node_id, is_primary_target=False, max_body_lines=5, max_methods_in_class_snippet=2)
                if rel_def_str:
                    item_tokens = estimate_tokens(rel_def_str)
                    if current_tokens + item_tokens <= max_context_tokens:
                        context_items_with_scores.append((score, rel_def_str)); current_tokens += item_tokens; processed_node_ids.add(rel_node_id)
                    else: break

        context_items_with_scores.sort(key=lambda x: x[0], reverse=True)
        final_context_str = "\n\n".join([item[1] for item in context_items_with_scores])
        return final_context_str if final_context_str else "No specific code context found based on the prompt."

if __name__ == '__main__':
    # ... (Example usage would require setting up a dummy project, loader, KG)
    print("ContextRetriever class with initial heuristics defined.")
