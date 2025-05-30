from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ImportInfo(BaseModel):
    name: str
    module: Optional[str] = None # For 'from module import name'
    asname: Optional[str] = None
    level: Optional[int] = None # For relative imports
    lineno: int
    # end_lineno: Optional[int] = None # Not always available for imports

class ArgumentInfo(BaseModel):
    name: str
    annotation: Optional[str] = None
    default_value: Optional[str] = None # Represent default value as string

class DecoratorInfo(BaseModel):
    name: str # The text of the decorator, e.g., "@staticmethod" or "@app.route('/path')"
    lineno: int

class BaseCodeElement(BaseModel):
    name: str
    lineno: int
    end_lineno: Optional[int] = None
    docstring: Optional[str] = None

class VariableInfo(BaseCodeElement):
    # Value is hard to represent generically and safely, focus on its existence and type if inferable
    # For now, just its name and location. Type inference can be added later.
    annotation: Optional[str] = None # Type hint if available

class FunctionInfo(BaseCodeElement):
    args: List[ArgumentInfo] = Field(default_factory=list)
    returns: Optional[str] = None # Return type annotation
    decorators: List[DecoratorInfo] = Field(default_factory=list)
    # Could add is_async if needed

class MethodInfo(FunctionInfo):
    # Inherits from FunctionInfo, can add specific things for methods if needed later
    # e.g., is_static, is_classmethod (though decorators can also show this)
    pass

class ClassInfo(BaseCodeElement):
    bases: List[str] = Field(default_factory=list) # List of base class names
    methods: List[MethodInfo] = Field(default_factory=list)
    class_variables: List[VariableInfo] = Field(default_factory=list) # Variables defined directly in class body
    decorators: List[DecoratorInfo] = Field(default_factory=list)

class ModuleInfo(BaseModel):
    module_name: str
    filepath: str
    language: Optional[str] = None # Language of the module (e.g., "python")
    docstring: Optional[str] = None # Module-level docstring
    imports: List[ImportInfo] = Field(default_factory=list)
    functions: List[FunctionInfo] = Field(default_factory=list)
    classes: List[ClassInfo] = Field(default_factory=list)
    variables: List[VariableInfo] = Field(default_factory=list) # Top-level variables in the module
    parse_errors: List[Dict[str, Any]] = Field(default_factory=list) # To store any errors from PythonParser

if __name__ == '__main__':
    print("Code Intelligence Models defined.")
