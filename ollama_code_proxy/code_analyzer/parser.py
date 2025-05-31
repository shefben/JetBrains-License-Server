import ast
import os
# import json # Not strictly needed for parser logic, but useful for testing/debugging in __main__
from typing import List, Dict, Any, Optional, Tuple, Union

from .models import (
    ArgumentInfo, DecoratorInfo, BaseCodeElement, VariableInfo,
    FunctionInfo, MethodInfo, ClassInfo, ModuleInfo, ImportInfo,
    CallInfo, CallArgumentInfo, InstanceCreationInfo, AttributeAccessInfo
)

class CodeParser:
    # Base class for future expansion to other languages.
    # For this phase, it's specific to Python's AST approach.
    def parse(self, filepath: str) -> Dict[str, Any]: # Return type changed to Dict
        raise NotImplementedError("Subclasses must implement this method.")

class IntraFunctionVisitor(ast.NodeVisitor):
    # Visitor to find calls, instantiations, and attribute accesses within a function/method.
    def __init__(self):
        self.function_calls: List[CallInfo] = []
        self.instance_creations: List[InstanceCreationInfo] = []
        self.attribute_accesses: List[AttributeAccessInfo] = []

    def _node_to_string(self, node: Optional[ast.AST]) -> str:
        if node is None:
            return "None"
        if hasattr(ast, 'unparse'): # Python 3.9+
            try:
                return ast.unparse(node)
            except Exception: # Fallback if unparse fails for some specific node types unexpectedly
                pass
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant): # Python 3.8+ for ast.Constant
            return repr(node.value)
        # For older Python versions (pre 3.8), ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Ellipsis
        if isinstance(node, (ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Ellipsis)): # type: ignore
             return repr(node.s if hasattr(node, 's') else (node.n if hasattr(node, 'n') else (node.value if hasattr(node, 'value') else '...'))) # type: ignore
        if isinstance(node, ast.Attribute):
            return f"{self._node_to_string(node.value)}.{node.attr}"

        # Fallback for complex nodes not easily representable as a simple string
        return f"complex_node:{type(node).__name__}"


    def visit_Call(self, node: ast.Call):
        target_name_str = self._node_to_string(node.func)

        args_info = [
            CallArgumentInfo(value_repr=self._node_to_string(arg))
            for arg in node.args
        ]
        keywords_info = [
            CallArgumentInfo(name=kw.arg, value_repr=self._node_to_string(kw.value))
            for kw in node.keywords if kw.arg is not None # kw.arg can be None for **kwargs
        ]

        is_likely_class_instantiation = False
        # Heuristic: if the callable part (node.func) is a Name or Attribute ending with an uppercase letter.
        # This is a basic heuristic and can be wrong (e.g. factory functions, classes not following CapWords).
        current_name_part = ""
        if isinstance(node.func, ast.Name):
            current_name_part = node.func.id
        elif isinstance(node.func, ast.Attribute):
            current_name_part = node.func.attr

        if current_name_part and current_name_part[0].isupper():
            is_likely_class_instantiation = True

        if is_likely_class_instantiation:
            self.instance_creations.append(InstanceCreationInfo(
                class_name=target_name_str,
                args=args_info,
                keywords=keywords_info,
                lineno=node.lineno
            ))
        else:
            self.function_calls.append(CallInfo(
                target_name=target_name_str,
                args=args_info,
                keywords=keywords_info,
                lineno=node.lineno
            ))

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)):
            self.attribute_accesses.append(AttributeAccessInfo(
                attribute_name=node.attr,
                target_object_name=self._node_to_string(node.value),
                lineno=node.lineno
            ))
        self.generic_visit(node)


class PythonParser(CodeParser):
    def __init__(self):
        pass

    def _get_docstring(self, node: Union[ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef]) -> Optional[str]:
        return ast.get_docstring(node, clean=False)

    def _parse_arguments(self, args_node: ast.arguments) -> List[ArgumentInfo]:
        parsed_args: List[ArgumentInfo] = []

        # Positional-only arguments (Python 3.8+)
        for arg in args_node.posonlyargs:
            parsed_args.append(ArgumentInfo(
                name=arg.arg,
                annotation=ast.unparse(arg.annotation) if hasattr(ast, 'unparse') and arg.annotation else None,
            ))

        # Regular arguments (positional or keyword)
        num_regular_args = len(args_node.args)
        num_defaults = len(args_node.defaults)
        defaults_offset = num_regular_args - num_defaults

        for i, arg in enumerate(args_node.args):
            default_value_str = None
            if i >= defaults_offset:
                default_node = args_node.defaults[i - defaults_offset]
                if default_node: default_value_str = ast.unparse(default_node) if hasattr(ast, 'unparse') else "..."

            parsed_args.append(ArgumentInfo(
                name=arg.arg,
                annotation=ast.unparse(arg.annotation) if hasattr(ast, 'unparse') and arg.annotation else None,
                default_value=default_value_str
            ))

        # *vararg
        if args_node.vararg:
            parsed_args.append(ArgumentInfo(
                name=f"*{args_node.vararg.arg}",
                annotation=ast.unparse(args_node.vararg.annotation) if hasattr(ast, 'unparse') and args_node.vararg.annotation else None
            ))

        # Keyword-only arguments
        for i, arg in enumerate(args_node.kwonlyargs):
            default_value_str = None
            if args_node.kw_defaults[i] is not None: # kw_defaults is a list, can have None for args without defaults
                default_node = args_node.kw_defaults[i]
                if default_node: default_value_str = ast.unparse(default_node) if hasattr(ast, 'unparse') else "..."

            parsed_args.append(ArgumentInfo(
                name=arg.arg,
                annotation=ast.unparse(arg.annotation) if hasattr(ast, 'unparse') and arg.annotation else None,
                default_value=default_value_str
            ))

        # **kwarg
        if args_node.kwarg:
            parsed_args.append(ArgumentInfo(
                name=f"**{args_node.kwarg.arg}",
                annotation=ast.unparse(args_node.kwarg.annotation) if hasattr(ast, 'unparse') and args_node.kwarg.annotation else None
            ))
        return parsed_args

    def _parse_decorators(self, decorator_list: List[ast.expr]) -> List[DecoratorInfo]:
        return [
            DecoratorInfo(name=ast.unparse(d) if hasattr(ast, 'unparse') else "unknown_decorator", lineno=d.lineno)
            for d in decorator_list
        ]

    def _is_method_static_or_class(self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Tuple[bool, bool]:
        is_static = False
        is_classmethod = False
        for dec_node in method_node.decorator_list:
            if isinstance(dec_node, ast.Name):
                if dec_node.id == 'staticmethod': is_static = True
                elif dec_node.id == 'classmethod': is_classmethod = True
            # More complex decorators (e.g. @foo.classmethod) require deeper inspection of `dec_node.attr`, etc.
        return is_static, is_classmethod

    def parse(self, filepath: str) -> Dict[str, Any]:
        module_name_str = os.path.splitext(os.path.basename(filepath))[0]
        analysis_result: Dict[str, Any] = {
            "module_name": module_name_str, "filepath": filepath, "language": "python",
            "docstring": None, "imports": [], "functions": [], "classes": [],
            "variables": [], "parse_errors": []
        }

        if not os.path.exists(filepath) or not filepath.endswith(".py"):
            analysis_result["parse_errors"].append({
                "type": "FileAccessError", "message": f"File not found or not a Python file: {filepath}"})
            return analysis_result

        try:
            with open(filepath, "r", encoding="utf-8", errors='ignore') as source_file:
                source_code = source_file.read()
            tree = ast.parse(source_code, filename=filepath) if source_code.strip() else ast.parse("", filename=filepath)
        except SyntaxError as e:
            analysis_result["parse_errors"].append({
                "type": "SyntaxError", "message": e.msg, "lineno": e.lineno,
                "offset": e.offset, "text": e.text})
            return analysis_result
        except Exception as e:
            analysis_result["parse_errors"].append({"type": "GeneralParsingError", "message": str(e)})
            return analysis_result

        analysis_result["docstring"] = self._get_docstring(tree)

        for node in tree.body:
            node_end_lineno = getattr(node, 'end_lineno', None)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis_result["imports"].append(ImportInfo(
                            name=alias.name, asname=alias.asname, lineno=node.lineno).model_dump())
                elif isinstance(node, ast.ImportFrom):
                    mod_name = node.module if node.module else "."
                    for alias in node.names:
                        analysis_result["imports"].append(ImportInfo(
                            module=mod_name, name=alias.name, asname=alias.asname,
                            level=node.level, lineno=node.lineno).model_dump())

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        analysis_result["variables"].append(VariableInfo(
                            name=target.id, lineno=node.lineno, end_lineno=node_end_lineno, annotation=None).model_dump())
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                 analysis_result["variables"].append(VariableInfo(
                    name=node.target.id, lineno=node.lineno, end_lineno=node_end_lineno,
                    annotation=ast.unparse(node.annotation) if hasattr(ast, 'unparse') and node.annotation else "...").model_dump())

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                intra_visitor = IntraFunctionVisitor()
                for stmt in node.body: intra_visitor.visit(stmt)

                analysis_result["functions"].append(FunctionInfo(
                    name=node.name, lineno=node.lineno, end_lineno=node_end_lineno,
                    docstring=self._get_docstring(node),
                    args=self._parse_arguments(node.args),
                    returns=ast.unparse(node.returns) if hasattr(ast, 'unparse') and node.returns else None,
                    decorators=self._parse_decorators(node.decorator_list),
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    function_calls=intra_visitor.function_calls,
                    instance_creations=intra_visitor.instance_creations,
                    attribute_accesses=intra_visitor.attribute_accesses
                ).model_dump())

            elif isinstance(node, ast.ClassDef):
                methods_list: List[MethodInfo] = [] # Ensure list of Pydantic models, not dicts
                class_vars_list: List[VariableInfo] = []
                for item in node.body:
                    item_end_lineno = getattr(item, 'end_lineno', None)
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        is_static, is_class = self._is_method_static_or_class(item)
                        intra_method_visitor = IntraFunctionVisitor()
                        for stmt_in_method in item.body: intra_method_visitor.visit(stmt_in_method)

                        methods_list.append(MethodInfo(
                            name=item.name, lineno=item.lineno, end_lineno=item_end_lineno,
                            docstring=self._get_docstring(item),
                            args=self._parse_arguments(item.args),
                            returns=ast.unparse(item.returns) if hasattr(ast, 'unparse') and item.returns else None,
                            decorators=self._parse_decorators(item.decorator_list),
                            is_async=isinstance(item, ast.AsyncFunctionDef),
                            is_static=is_static, is_classmethod=is_class,
                            function_calls=intra_method_visitor.function_calls,
                            instance_creations=intra_method_visitor.instance_creations,
                            attribute_accesses=intra_method_visitor.attribute_accesses
                        ))
                    elif isinstance(item, ast.Assign):
                        for target_node in item.targets: # Corrected variable name
                            if isinstance(target_node, ast.Name):
                                class_vars_list.append(VariableInfo(name=target_node.id, lineno=item.lineno, end_lineno=item_end_lineno))
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        class_vars_list.append(VariableInfo(
                            name=item.target.id, lineno=item.lineno, end_lineno=item_end_lineno,
                            annotation=ast.unparse(item.annotation) if hasattr(ast, 'unparse') and item.annotation else "..."
                        ))

                analysis_result["classes"].append(ClassInfo(
                    name=node.name, lineno=node.lineno, end_lineno=node_end_lineno,
                    docstring=self._get_docstring(node),
                    bases=[ast.unparse(b) if hasattr(ast, 'unparse') else "base_class" for b in node.bases],
                    methods=methods_list,
                    class_variables=class_vars_list,
                    decorators=self._parse_decorators(node.decorator_list)
                ).model_dump())

        return analysis_result

if __name__ == '__main__':
    # Example usage for direct testing of the parser
    import json # Moved import here as it's only for the example
    print("Running PythonParser example...")
    parser = PythonParser()

    dummy_file_content = """
import os
from typing import List

GLOBAL_VAR: int = 100 # Top-level variable

class MyHelperClass:
    '''Helper class docstring'''
    def helper_method(self): # Method
        return "helper"

@some_decorator # Class decorator
class MyClass(object): # Class definition
    '''A test class.'''
    cls_var: str = "test" # Class variable

    def __init__(self, name: str = "default"): # Method with args and defaults
        self.name = name # Attribute access (store)
        self.helper = MyHelperClass() # Instance creation
        self.value = self.helper.helper_method() # Attribute access (load) and call
        local_call = print("Setup complete") # Call

    @staticmethod
    def static_method(x, y=10): # Static method
        '''A static method.'''
        res = x + y
        print(f"Static result: {res}") # Call
        return res

    async def async_method(self, items: List[str]): # Async method
        for item in items:
            processed_item = self._process_item(item) # Call
            print(item) # Call
        return GLOBAL_VAR # Attribute access (load global)

    def _process_item(self, item: str) -> str: # Private-like method
        return item.upper() # Attribute access (str method call)
"""
    dummy_filepath = "dummy_parser_test_file.py"
    with open(dummy_filepath, "w", encoding="utf-8") as f:
        f.write(dummy_file_content)

    parsed_module_data = parser.parse(dummy_filepath)

    if parsed_module_data and not parsed_module_data.get('parse_errors'):
        print(f"Parsed module: {parsed_module_data.get('module_name')}")
        # Full dump can be very verbose, uncomment if detailed inspection is needed
        # print(json.dumps(parsed_module_data, indent=2))

        print(f"\nModule Docstring: {parsed_module_data.get('docstring')}")

        for imp_info in parsed_module_data.get('imports', []):
            print(f"  Import: name='{imp_info.get('name')}', module='{imp_info.get('module')}', asname='{imp_info.get('asname')}'")

        for var_info in parsed_module_data.get('variables', []):
            print(f"  Variable: name='{var_info.get('name')}', annotation='{var_info.get('annotation')}'")

        for func_info_dict in parsed_module_data.get('functions', []): # Top-level functions (none in this example)
            # Deserialize back to model for easier access if needed, or access as dict
            func_info = FunctionInfo(**func_info_dict) # type: ignore
            print(f"  Function: {func_info.name}")

        for cls_info_dict in parsed_module_data.get('classes', []):
            cls_info = ClassInfo(**cls_info_dict) # type: ignore
            print(f"  Class: {cls_info.name} (bases: {cls_info.bases})")
            if cls_info.decorators: print(f"    Decorators: {[d.name for d in cls_info.decorators]}")
            for class_var_info_obj in cls_info.class_variables: # class_variables is List[VariableInfo]
                 print(f"    Class Var: name='{class_var_info_obj.name}', annotation='{class_var_info_obj.annotation}'")
            for meth_info_obj in cls_info.methods: # methods is List[MethodInfo]
                print(f"    Method: {meth_info_obj.name} (async: {meth_info_obj.is_async}, static: {meth_info_obj.is_static}, classmethod: {meth_info_obj.is_classmethod})")
                if meth_info_obj.args: print(f"      Args: {[a.name for a in meth_info_obj.args]}")
                if meth_info_obj.function_calls: print(f"      Calls: {[c.target_name for c in meth_info_obj.function_calls]}")
                if meth_info_obj.instance_creations: print(f"      Instances: {[i.class_name for i in meth_info_obj.instance_creations]}")
                if meth_info_obj.attribute_accesses: print(f"      Attrs: {[ (a.target_object_name + '.' + a.attribute_name) if a.target_object_name else a.attribute_name for a in meth_info_obj.attribute_accesses]}")

    elif parsed_module_data and parsed_module_data.get('parse_errors'):
        print(f"Parsing errors for {dummy_filepath}: {parsed_module_data.get('parse_errors')}")
    else:
        print(f"Failed to parse {dummy_filepath} - no data returned.")

    if os.path.exists(dummy_filepath):
        os.remove(dummy_filepath)
    print("PythonParser example finished.")
