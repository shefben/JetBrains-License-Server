from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Union

# --- Base and Common Info Models ---
class ArgumentInfo(BaseModel):
    name: str
    annotation: Optional[str] = None
    default_value: Optional[str] = None

class DecoratorInfo(BaseModel):
    name: str
    lineno: int

class BaseCodeElement(BaseModel):
    name: str
    lineno: int
    end_lineno: Optional[int] = None
    docstring: Optional[str] = None

# --- Models for Reference Following (Intra-File and Cross-File) ---
class CallArgumentInfo(BaseModel):
    name: Optional[str] = None
    value_repr: str
    # resolved_value_type: Optional[str] = None # Future: for basic type inference

class CallInfo(BaseModel):
    target_name: str
    args: List[CallArgumentInfo] = Field(default_factory=list)
    keywords: List[CallArgumentInfo] = Field(default_factory=list)
    lineno: int
    # New fields for cross-file resolution:
    resolved_target_filepath: Optional[str] = None # Filepath of the module where the target is defined
    resolved_target_name: Optional[str] = None     # Fully qualified name of the target (e.g., ClassName.method or function_name)
    is_method_call: bool = False
    # called_on_variable_name: Optional[str] = None

class InstanceCreationInfo(BaseModel):
    class_name: str
    args: List[CallArgumentInfo] = Field(default_factory=list)
    keywords: List[CallArgumentInfo] = Field(default_factory=list)
    lineno: int
    # New fields for cross-file resolution:
    resolved_target_filepath: Optional[str] = None # Filepath of the module where the class is defined

class AttributeAccessInfo(BaseModel):
    attribute_name: str
    target_object_name: Optional[str]
    lineno: int
    # context: str
    # resolved_target_filepath: Optional[str] = None
    # resolved_target_type: Optional[str] = None

# --- Existing Info Models (Modified to include new reference info) ---
class ImportInfo(BaseModel):
    name: str
    module: Optional[str] = None
    asname: Optional[str] = None
    level: Optional[int] = None
    lineno: int
    end_lineno: Optional[int] = None # Not reliably available from AST
    # New fields for resolved import:
    resolved_filepath: Optional[str] = None # Absolute or project-relative filepath to the .py file this import points to

class VariableInfo(BaseCodeElement):
    annotation: Optional[str] = None

class FunctionInfo(BaseCodeElement):
    args: List[ArgumentInfo] = Field(default_factory=list)
    returns: Optional[str] = None
    decorators: List[DecoratorInfo] = Field(default_factory=list)
    is_async: bool = False
    function_calls: List[CallInfo] = Field(default_factory=list)
    instance_creations: List[InstanceCreationInfo] = Field(default_factory=list)
    attribute_accesses: List[AttributeAccessInfo] = Field(default_factory=list)

class MethodInfo(FunctionInfo):
    is_static: bool = False
    is_classmethod: bool = False

class ClassInfo(BaseCodeElement):
    bases: List[str] = Field(default_factory=list)
    methods: List[MethodInfo] = Field(default_factory=list)
    class_variables: List[VariableInfo] = Field(default_factory=list)
    decorators: List[DecoratorInfo] = Field(default_factory=list)
    # New field for resolved base classes
    resolved_bases: List[Dict[str, Optional[str]]] = Field(default_factory=list) # List of {"name": "BaseClassName", "filepath": "/path/to/base.py" or None if not resolved}


class ModuleInfo(BaseModel):
    module_name: str
    filepath: str # Relative path within the analyzed codebase
    language: Optional[str] = "python"
    docstring: Optional[str] = None
    imports: List[ImportInfo] = Field(default_factory=list)
    functions: List[FunctionInfo] = Field(default_factory=list)
    classes: List[ClassInfo] = Field(default_factory=list)
    variables: List[VariableInfo] = Field(default_factory=list)
    parse_errors: List[Dict[str, Any]] = Field(default_factory=list)
