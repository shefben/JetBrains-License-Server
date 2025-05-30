import ast
from typing import List, Dict, Any

class CodeParser:
    """
    Base class for code parsers.
    Each language-specific parser should inherit from this class.
    """
    def __init__(self, language: str):
        self.language = language

    def parse(self, code: str) -> Dict[str, Any]:
        """
        Parses the given code string and returns a structured representation.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class PythonParser(CodeParser):
    """
    Parser for Python code.
    """
    def __init__(self):
        super().__init__("python")

    def parse(self, code: str) -> Dict[str, Any]:
        """
        Parses Python code using the ast module.
        Extracts functions, classes, imports, and module-level docstring.
        """
        analysis = {
            "language": self.language,
            "docstring": None,
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [], # For top-level variable assignments
            "parse_errors": []
        }
        try:
            tree = ast.parse(code)
            analysis["docstring"] = ast.get_docstring(tree, clean=False) # Get module docstring

            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno
                    })
                elif isinstance(node, ast.ClassDef):
                    class_details = {
                        "name": node.name,
                        "methods": [],
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno
                    }
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef): # Methods in a class
                            class_details["methods"].append({
                                "name": item.name,
                                "args": [arg.arg for arg in item.args.args],
                                "lineno": item.lineno,
                                "end_lineno": item.end_lineno
                            })
                    analysis["classes"].append(class_details)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append({
                            "name": alias.name,
                            "asname": alias.asname,
                            "lineno": node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    analysis["imports"].append({
                        "module": node.module,
                        "names": [{"name": alias.name, "asname": alias.asname} for alias in node.names],
                        "level": node.level,
                        "lineno": node.lineno
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name): # Simple variable assignment
                            analysis["variables"].append({
                                "name": target.id,
                                "lineno": node.lineno,
                                "end_lineno": node.end_lineno
                            })
        except SyntaxError as e:
            analysis["parse_errors"].append({
                "type": "SyntaxError",
                "message": e.msg,
                "lineno": e.lineno,
                "offset": e.offset,
                "text": e.text
            })
        except Exception as e:
            analysis["parse_errors"].append({
                "type": "GeneralParsingError",
                "message": str(e)
            })
        return analysis

# Example usage:
if __name__ == "__main__":
    python_code_example = """
import os
import sys as system_alias

from math import sqrt, pow as power

class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

def my_function(a, b):
    x = a + b # A local variable
    return x

another_var = 100
"""
    parser = PythonParser()
    parsed_ast = parser.parse(python_code_example)
    print("--- Parsed Python Code ---")
    import json
    print(json.dumps(parsed_ast, indent=2))

    error_code_example = """
def my_func(
    print("hello")
"""
    parser = PythonParser() # Re-use parser instance
    parsed_error_ast = parser.parse(error_code_example)
    print("\\n--- Parsed Python Code with Syntax Error ---")
    print(json.dumps(parsed_error_ast, indent=2))

    empty_code_example = ""
    parsed_empty_ast = parser.parse(empty_code_example) # Re-use parser instance
    print("\\n--- Parsed Empty Python Code ---")
    print(json.dumps(parsed_empty_ast, indent=2))

    # Example of code that parses fine but has no specific elements like functions or classes
    just_a_variable = "a = 1"
    parsed_just_var = parser.parse(just_a_variable)
    print("\\n--- Parsed Python Code with just a variable ---")
    print(json.dumps(parsed_just_var, indent=2))
