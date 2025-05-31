import os
import sys
from typing import Dict, List, Optional, Tuple, Set, Union # Added Union

from .models import (
    ModuleInfo, ImportInfo, FunctionInfo, MethodInfo, ClassInfo,
    CallInfo, InstanceCreationInfo, AttributeAccessInfo # Added AttributeAccessInfo
)


class ReferenceResolver:
    def __init__(self, all_modules: Dict[str, ModuleInfo], codebase_root: str):
        self.all_modules: Dict[str, ModuleInfo] = all_modules  # filepath (abs) -> ModuleInfo
        self.codebase_root: str = os.path.abspath(codebase_root)
        self._resolved_module_cache: Dict[Tuple[str, str, int], Optional[str]] = {}
        self._module_symbol_tables: Dict[str, Dict[str, str]] = {} # filepath (abs) -> {symbol_name: type}

    def _build_module_symbol_tables(self):
        for abs_filepath, module_info in self.all_modules.items():
            symbols: Dict[str, str] = {}
            for func_info in module_info.functions:
                symbols[func_info.name] = "function"
            for class_info in module_info.classes:
                symbols[class_info.name] = "class"
            self._module_symbol_tables[abs_filepath] = symbols

    def _normalize_path(self, path: str) -> str:
        return os.path.normpath(os.path.abspath(path))

    def _find_module_path_from_import(
        self,
        importer_module_abs_filepath: str,
        import_name_parts: List[str],
        level: int
    ) -> Optional[str]: # Returns absolute filepath

        importer_dir = os.path.dirname(importer_module_abs_filepath)
        cache_key = (importer_dir, ".".join(import_name_parts), level)
        if cache_key in self._resolved_module_cache:
            return self._resolved_module_cache[cache_key]

        base_search_path = ""
        if level == 0:
            base_search_path = self.codebase_root
        elif level > 0:
            current_path = importer_dir
            for _ in range(level - 1):
                current_path = os.path.dirname(current_path)
            base_search_path = current_path
        else:
            self._resolved_module_cache[cache_key] = None
            return None

        potential_module_path_py = os.path.join(base_search_path, *import_name_parts) + ".py"
        normalized_path_py = self._normalize_path(potential_module_path_py)
        if normalized_path_py in self.all_modules: # Check against keys of all_modules (abs paths)
            self._resolved_module_cache[cache_key] = normalized_path_py
            return normalized_path_py

        potential_package_init_path = os.path.join(base_search_path, *import_name_parts, "__init__.py")
        normalized_package_init_path = self._normalize_path(potential_package_init_path)
        if normalized_package_init_path in self.all_modules:
            self._resolved_module_cache[cache_key] = normalized_package_init_path
            return normalized_package_init_path

        # Simplified fallback for absolute imports if not found directly under codebase_root
        # (e.g. if codebase_root is /project and import is my_package.utils)
        if level == 0 and import_name_parts:
            # This checks if the path formed by joining codebase_root and import_name_parts exists
            # This is somewhat redundant with the first check if base_search_path was already codebase_root
            # but can act as a secondary check or for slightly different path structures.
            # For a robust solution, Python's import algorithm or a similar one needs to be mimicked.
            pass # Current logic is primarily based on self.all_modules keys containing abs paths.

        self._resolved_module_cache[cache_key] = None
        return None

    def _resolve_imports_for_module(self, module_info: ModuleInfo):
        # module_info.filepath is relative to codebase_root
        # We need its absolute path to correctly resolve relative imports from it.
        importer_abs_filepath = self._normalize_path(os.path.join(self.codebase_root, module_info.filepath))

        for imp_info in module_info.imports:
            if imp_info.resolved_filepath: continue

            resolved_module_abs_file: Optional[str] = None
            if imp_info.module:
                import_name_parts = imp_info.module.split('.')
                resolved_module_abs_file = self._find_module_path_from_import(
                    importer_abs_filepath, import_name_parts, imp_info.level or 0
                )
            else:
                import_name_parts = imp_info.name.split('.')
                resolved_module_abs_file = self._find_module_path_from_import(
                    importer_abs_filepath, import_name_parts, 0
                )

            if resolved_module_abs_file:
                # Store path relative to codebase_root for consistency with ModuleInfo.filepath
                imp_info.resolved_filepath = os.path.relpath(resolved_module_abs_file, self.codebase_root)


    def _find_symbol_definition(
        self,
        scope_module_info: ModuleInfo,
        symbol_name_parts: List[str]
    ) -> Optional[Tuple[str, str, str]]: # (relative_filepath, symbol_name_in_that_file, type)

        if not symbol_name_parts: return None

        scope_module_abs_filepath = self._normalize_path(os.path.join(self.codebase_root, scope_module_info.filepath))

        if len(symbol_name_parts) == 1:
            target_name = symbol_name_parts[0]
            if target_name in self._module_symbol_tables.get(scope_module_abs_filepath, {}):
                symbol_type = self._module_symbol_tables[scope_module_abs_filepath][target_name]
                return scope_module_info.filepath, target_name, symbol_type # Return relative path

            for imp_info in scope_module_info.imports:
                original_imported_name = imp_info.name # The actual name of the symbol from its source module
                effective_name_in_scope = imp_info.asname or imp_info.name # How it's known in current scope

                if effective_name_in_scope == target_name:
                    if imp_info.resolved_filepath: # This is relative path
                        resolved_abs_filepath = self._normalize_path(os.path.join(self.codebase_root, imp_info.resolved_filepath))
                        imported_module_symbols = self._module_symbol_tables.get(resolved_abs_filepath, {})

                        if imp_info.module: # 'from module import symbol_name' (original_imported_name is symbol_name)
                            if original_imported_name in imported_module_symbols:
                                symbol_type = imported_module_symbols[original_imported_name]
                                return imp_info.resolved_filepath, original_imported_name, symbol_type
                        else: # 'import module' or 'import module as alias' (original_imported_name is the module name)
                              # This case implies target_name is an alias for a module.
                              # If we are looking for a function/class by this name, it's not here.
                              # This branch is if 'target_name' is an alias for a *symbol* within the module,
                              # which is covered by the `imp_info.module` case.
                              # If `import module` and `target_name` is `module`, this means we are trying to call a module.
                              # This should be handled by the len > 1 case.
                              pass
            return None

        base_name = symbol_name_parts[0]
        member_name_str = ".".join(symbol_name_parts[1:]) # e.g. "ClassName" or "sub_member.another_member"

        resolved_base_abs_filepath: Optional[str] = None

        for imp_info in scope_module_info.imports:
            # Handles 'import module as base_name' or 'import package.module as base_name'
            # Or 'import base_name' (where base_name is a module)
            # Or 'from package import base_name' (where base_name is a module)
            is_direct_module_import = not imp_info.module and (imp_info.asname == base_name or (imp_info.asname is None and imp_info.name.split('.')[-1] == base_name))
            is_from_import_module = imp_info.module and (imp_info.asname == base_name or (imp_info.asname is None and imp_info.name == base_name))

            if is_direct_module_import or is_from_import_module:
                if imp_info.resolved_filepath: # This is relative path
                    resolved_base_abs_filepath = self._normalize_path(os.path.join(self.codebase_root, imp_info.resolved_filepath))
                    break

        if resolved_base_abs_filepath:
            # For now, assume member_name_str is a top-level symbol in the resolved_base_abs_filepath
            # Does not handle class methods or attributes within classes yet (e.g. MyClass.static_method)
            # or nested modules (e.g. my_pkg.sub_module.func)
            imported_module_symbols = self._module_symbol_tables.get(resolved_base_abs_filepath, {})
            if member_name_str in imported_module_symbols:
                symbol_type = imported_module_symbols[member_name_str]
                return os.path.relpath(resolved_base_abs_filepath, self.codebase_root), member_name_str, symbol_type

        return None

    def _resolve_references_in_module_elements(self, module_info: ModuleInfo):
        elements_to_process: List[Union[FunctionInfo, MethodInfo]] = [] # type: ignore
        elements_to_process.extend(module_info.functions)
        for class_info in module_info.classes:
            elements_to_process.extend(class_info.methods)

        for item in elements_to_process:
            for call_info in item.function_calls:
                if call_info.resolved_target_filepath: continue
                name_parts = call_info.target_name.split('.')

                if name_parts[0] in ["self", "cls"] and len(name_parts) > 1:
                    call_info.is_method_call = True
                    # TODO: Resolve method calls on self/cls. Needs class hierarchy info.
                    continue

                resolved = self._find_symbol_definition(module_info, name_parts)
                if resolved:
                    call_info.resolved_target_filepath = resolved[0] # Relative path
                    call_info.resolved_target_name = resolved[1]

            for inst_info in item.instance_creations:
                if inst_info.resolved_target_filepath: continue
                name_parts = inst_info.class_name.split('.')
                resolved = self._find_symbol_definition(module_info, name_parts)
                if resolved and resolved[2] == 'class':
                    inst_info.resolved_target_filepath = resolved[0] # Relative path

        for class_info in module_info.classes:
            class_info.resolved_bases = []
            for base_name_str in class_info.bases:
                name_parts = base_name_str.split('.')
                resolved = self._find_symbol_definition(module_info, name_parts)
                if resolved and resolved[2] == 'class':
                    class_info.resolved_bases.append({"name": base_name_str, "filepath": resolved[0]}) # Relative path
                else:
                    class_info.resolved_bases.append({"name": base_name_str, "filepath": None})

    def resolve_all_references(self):
        self._build_module_symbol_tables() # Uses absolute filepaths as keys

        # Pass module_info (which has relative filepath) to _resolve_imports_for_module
        for module_info in self.all_modules.values():
            self._resolve_imports_for_module(module_info)

        # Pass module_info to _resolve_references_in_module_elements
        for module_info in self.all_modules.values():
            self._resolve_references_in_module_elements(module_info)

if __name__ == '__main__':
    print("ReferenceResolver class defined. Example usage would require mock ModuleInfo data.")
