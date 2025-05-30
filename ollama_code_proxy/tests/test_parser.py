import pytest
import json
from code_analyzer.parser import PythonParser, CodeParser

class TestCodeParser:
    def test_base_parser_raises_not_implemented(self):
        parser = CodeParser("test_lang")
        with pytest.raises(NotImplementedError):
            parser.parse("some code")

class TestPythonParser:
    @pytest.fixture
    def parser(self):
        return PythonParser()

    def test_parse_empty_code(self, parser):
        result = parser.parse("")
        assert result["language"] == "python"
        assert result["functions"] == []
        assert result["classes"] == []
        assert result["imports"] == []
        assert result["variables"] == []

    def test_parse_simple_function(self, parser):
        code = """
def greet(name):
    print(f"Hello, {name}!")
"""
        result = parser.parse(code)
        assert len(result["functions"]) == 1
        func = result["functions"][0]
        assert func["name"] == "greet"
        assert func["args"] == ["name"]
        assert func["lineno"] is not None
        assert func["end_lineno"] is not None


    def test_parse_simple_class(self, parser):
        code = """
class Greeter:
    def __init__(self, greeting):
        self.greeting = greeting

    def greet(self, name):
        return f"{self.greeting}, {name}!"
"""
        result = parser.parse(code)
        assert len(result["classes"]) == 1
        cls = result["classes"][0]
        assert cls["name"] == "Greeter"
        assert len(cls["methods"]) == 2
        assert cls["methods"][0]["name"] == "__init__"
        assert cls["methods"][0]["args"] == ["self", "greeting"]
        assert cls["methods"][1]["name"] == "greet"
        assert cls["methods"][1]["args"] == ["self", "name"]
        assert cls["lineno"] is not None
        assert cls["end_lineno"] is not None


    def test_parse_imports(self, parser):
        code = """
import os
import sys as system
from math import sqrt, pow as power
from . import local_module
from ..parent_module import another_item
"""
        result = parser.parse(code)
        assert len(result["imports"]) == 5

        # Basic import
        assert any(imp["name"] == "os" and imp["asname"] is None for imp in result["imports"])
        # Import with alias
        assert any(imp["name"] == "sys" and imp["asname"] == "system" for imp in result["imports"])
        # From import with multiple names and one alias
        from_math = next(imp for imp in result["imports"] if imp.get("module") == "math")
        assert {"name": "sqrt", "asname": None} in from_math["names"]
        assert {"name": "pow", "asname": "power"} in from_math["names"]

        # Relative imports
        assert any(imp.get("module") is None and imp.get("level") == 1 and imp["names"][0]["name"] == "local_module" for imp in result["imports"])
        assert any(imp.get("module") == "parent_module" and imp.get("level") == 2 and imp["names"][0]["name"] == "another_item" for imp in result["imports"])


    def test_parse_top_level_variables(self, parser):
        code = """
MY_CONSTANT = 100
another_variable = "hello"
is_active = True
"""
        result = parser.parse(code)
        assert len(result["variables"]) == 3
        assert any(var["name"] == "MY_CONSTANT" and var["lineno"] is not None for var in result["variables"])
        assert any(var["name"] == "another_variable" and var["lineno"] is not None for var in result["variables"])
        assert any(var["name"] == "is_active" and var["lineno"] is not None for var in result["variables"])

    def test_parse_syntax_error(self, parser):
        code = "def greet(name:\n    print(f\"Hello, {name}!\")"
        result = parser.parse(code)
        assert "error" in result
        assert "Syntax error" in result["error"]
        assert result["lineno"] is not None
        assert result["offset"] is not None
        assert result["text"] is not None

    def test_parse_function_with_docstring(self, parser):
        code = '''
def my_func():
    """This is a docstring."""
    pass
'''
        result = parser.parse(code)
        assert len(result["functions"]) == 1
        assert result["functions"][0]["name"] == "my_func"

    def test_parse_class_with_docstring(self, parser):
        code = '''
class MyClass:
    """This is a class docstring."""
    def method(self):
        """This is a method docstring."""
        pass
'''
        result = parser.parse(code)
        assert len(result["classes"]) == 1
        assert result["classes"][0]["name"] == "MyClass"
        assert len(result["classes"][0]["methods"]) == 1
        assert result["classes"][0]["methods"][0]["name"] == "method"

    def test_parse_code_with_comments(self, parser):
        code = """
# This is a full line comment
def func_with_comment(): # This is an inline comment
    # Another comment inside
    pass
"""
        result = parser.parse(code)
        # Comments are generally ignored by ast.parse unless special comment parsing is used
        # So we just check if the function is parsed correctly
        assert len(result["functions"]) == 1
        assert result["functions"][0]["name"] == "func_with_comment"

    def test_end_lineno_for_nodes(self, parser):
        code = """
def my_func(a): # line 2
    return a # line 3

class MyClazz: # line 5
    pass # line 6

var = 10 # line 8
"""
        # Python 3.8+ provides end_lineno for many node types
        result = parser.parse(code)
        assert result["functions"][0]["name"] == "my_func"
        assert result["functions"][0]["lineno"] == 2
        assert result["functions"][0]["end_lineno"] is not None

        assert result["classes"][0]["name"] == "MyClazz"
        assert result["classes"][0]["lineno"] == 5
        assert result["classes"][0]["end_lineno"] is not None

        assert result["variables"][0]["name"] == "var"
        assert result["variables"][0]["lineno"] == 8
        assert result["variables"][0]["end_lineno"] is not None # ast.Assign has end_lineno in Py3.8+

        # For debugging if a specific test fails on end_lineno
        # print(json.dumps(result, indent=2))
