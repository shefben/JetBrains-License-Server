from pydantic import BaseModel, Field # Removed RootModel as it's not used
from typing import Dict, List, Optional, Tuple, Any
import random
import copy # For deepcopying domain_heuristic_strengths

# --- PromptProgram Definition ---

class PromptProgram(BaseModel):
    max_primary_symbols: int = Field(default=2, ge=1, le=5, description="Max primary symbols from prompt to focus on.")

    include_callees: bool = Field(default=True)
    max_callees_to_include: int = Field(default=2, ge=0, le=5)
    callee_snippet_max_lines: int = Field(default=10, ge=0, le=30, description="0 means signature only")

    include_callers: bool = Field(default=False)
    max_callers_to_include: int = Field(default=1, ge=0, le=3)
    caller_snippet_max_lines: int = Field(default=5, ge=0, le=20)

    include_inheritance_parents: bool = Field(default=True)
    max_inheritance_depth: int = Field(default=1, ge=0, le=3)
    parent_class_snippet_max_lines: int = Field(default=15, ge=0, le=40)

    include_inheritance_children: bool = Field(default=False)
    max_children_to_include: int = Field(default=1, ge=0, le=3)

    primary_symbol_snippet_max_lines: int = Field(default=30, ge=5, le=100)

    instructional_prefix_key: str = Field(default="default_v1", description="Key to select a predefined instructional prefix.")

    domain_heuristic_strengths: Dict[str, float] = Field(default_factory=lambda: {
        "database": 0.5, "api": 0.5, "test": 0.7, "ui": 0.3, "data_processing": 0.4,
        "concurrency": 0.4, "file_io": 0.3, "security": 0.6, "logging": 0.3
    })

    class Config:
        validate_assignment = True

# --- PromptProgramGenerator Definition ---

class PromptProgramGenerator:
    def __init__(self, predefined_instructional_prefixes: Optional[Dict[str, str]] = None):
        self.predefined_instructional_prefixes = predefined_instructional_prefixes or {
            "default_v1": (
                "You are an expert Python programming assistant. The user is asking a question about a codebase.\n"
                "Use the following provided code context to understand the relevant parts of the codebase.\n"
                "Based on this context AND the user's request, provide a comprehensive and accurate response.\n"
                "If the context is insufficient, state that and try to answer based on general knowledge if appropriate.\n"
                "Do not refer to 'the context provided' in your answer, just use it."
            ),
            "completion_v1": (
                "You are a code completion engine. Complete the following Python code based on the provided context.\n"
                "Only output the completed code block. No explanations."
            ),
            "debug_v1": (
                "You are a debugging assistant. Analyze the provided code context and the user's problem description.\n"
                "Identify potential causes for the issue and suggest debugging steps or fixes.\n"
                "Consider edge cases and common pitfalls related to the context."
            )
        }
        self.prefix_keys = list(self.predefined_instructional_prefixes.keys())
        # Get available domain keys from PromptProgram's default factory
        self.domain_keys = list(PromptProgram().domain_heuristic_strengths.keys())

    def generate_default_program(self) -> PromptProgram:
        return PromptProgram()

    def _get_field_constraints(self, field_name: str) -> Tuple[Any, Optional[Union[int, float]], Optional[Union[int, float]]]: # type: ignore
        # Helper to attempt to get ge/le constraints. Pydantic v2 makes this complex.
        # This is a simplified version. For robust constraint extraction, refer to Pydantic docs.
        field_info = PromptProgram.model_fields.get(field_name)
        if not field_info: return None, None, None

        default_val = field_info.default
        ge_val, le_val = None, None

        # In Pydantic V2, constraints are often in `metadata` or part of the `PydanticAnnotation`
        # This simplified approach checks for 'ge' and 'le' attributes if they were set via Field(ge=X, le=Y)
        # This might not capture all ways constraints can be defined.
        if field_info.json_schema_extra: # Check if extra attributes were passed to Field
             if isinstance(field_info.json_schema_extra, dict):
                ge_val = field_info.json_schema_extra.get('ge')
                le_val = field_info.json_schema_extra.get('le')

        # For pydantic.Field instances directly as defaults (less common for simple types)
        if isinstance(default_val, Field) and hasattr(default_val, 'default'):
            field_instance_info = default_val # It's actually the FieldInfo object
            default_val = field_instance_info.default
            if hasattr(field_instance_info, 'ge') and field_instance_info.ge is not None: ge_val = field_instance_info.ge
            if hasattr(field_instance_info, 'le') and field_instance_info.le is not None: le_val = field_instance_info.le

        return default_val, ge_val, le_val

    def generate_random_program(self) -> PromptProgram:
        params = {}
        fallback_ranges_int = {
            "max_primary_symbols": (1,3), "max_callees_to_include": (0,3),
            "callee_snippet_max_lines": (0,20), "max_callers_to_include": (0,2),
            "caller_snippet_max_lines": (0,15), "max_inheritance_depth": (0,2),
            "parent_class_snippet_max_lines": (0,25), "max_children_to_include": (0,1),
            "primary_symbol_snippet_max_lines": (10, 50),
        }

        for field_name, field_info in PromptProgram.model_fields.items():
            actual_type = field_info.annotation
            _default_val, ge_val, le_val = self._get_field_constraints(field_name)

            if actual_type is int:
                min_r, max_r = fallback_ranges_int.get(field_name, (0, 10))
                min_val = ge_val if ge_val is not None else min_r
                max_val = le_val if le_val is not None else max_r
                params[field_name] = random.randint(int(min_val), int(max_val))
            elif actual_type is bool:
                params[field_name] = random.choice([True, False])
            elif field_name == "instructional_prefix_key":
                params[field_name] = random.choice(self.prefix_keys)
            elif field_name == "domain_heuristic_strengths":
                strengths = {}
                for domain_key in self.domain_keys:
                    if random.random() < 0.7:
                         strengths[domain_key] = round(random.uniform(0.0, 1.0), 2)
                params[field_name] = strengths
            elif field_info.default is not None:
                params[field_name] = field_info.default
            elif field_info.default_factory is not None:
                 params[field_name] = field_info.default_factory()


        return PromptProgram(**params)

    def mutate_program(self, program: PromptProgram, mutation_probability: float = 0.2, mutation_strength: float = 0.3) -> PromptProgram:
        mutated_params = program.model_dump()

        for field_name, field_info in PromptProgram.model_fields.items():
            if random.random() < mutation_probability:
                current_value = mutated_params[field_name]
                actual_type = field_info.annotation
                _default_val, ge_val, le_val = self._get_field_constraints(field_name)

                if actual_type is int:
                    min_val = int(ge_val if ge_val is not None else 0)
                    max_val_est = current_value + 10
                    if field_name in ["max_primary_symbols","max_callees_to_include","max_callers_to_include","max_inheritance_depth","max_children_to_include"]:
                        max_val_est = 5
                    elif "snippet_max_lines" in field_name: max_val_est = 50

                    max_val = int(le_val if le_val is not None else max_val_est)

                    range_val = max_val - min_val
                    if range_val <=0 : range_val = 10

                    change = int(range_val * mutation_strength * random.choice([-1, 1]))
                    if change == 0 and max_val > min_val : change = random.choice([-1,1])

                    new_value = current_value + change
                    mutated_params[field_name] = max(min_val, min(new_value, max_val))

                elif actual_type is bool:
                    mutated_params[field_name] = not current_value

                elif field_name == "instructional_prefix_key":
                    mutated_params[field_name] = random.choice([k for k in self.prefix_keys if k != current_value] or self.prefix_keys)

                elif field_name == "domain_heuristic_strengths":
                    new_strengths = copy.deepcopy(current_value) # current_value is a dict
                    # Mutate existing strengths
                    for domain_key in list(new_strengths.keys()): # Iterate over copy of keys if removing
                        if random.random() < 0.5: # Mutate strength of this domain
                             change = 0.4 * mutation_strength * random.uniform(-1,1) # Use mutation_strength for amount
                             new_strengths[domain_key] = round(max(0.0, min(1.0, new_strengths[domain_key] + change)),2)
                        elif random.random() < 0.1 and len(new_strengths) > 1: # Small chance to remove a domain
                            del new_strengths[domain_key]
                    # Potentially add a new domain key
                    if self.domain_keys and random.random() < 0.2:
                        available_to_add = [dk for dk in self.domain_keys if dk not in new_strengths]
                        if available_to_add: new_strengths[random.choice(available_to_add)] = round(random.uniform(0.1,0.9),2)
                    mutated_params[field_name] = new_strengths
        try:
            return PromptProgram(**mutated_params)
        except Exception as e: # Fallback if mutation leads to invalid state not caught by clamps
            # print(f"Warning: Mutation resulted in invalid PromptProgram, returning original. Error: {e}")
            return program # Return original if mutation fails validation

if __name__ == '__main__':
    import json
    print("PromptProgram and PromptProgramGenerator defined.")

    generator = PromptProgramGenerator()

    default_prog = generator.generate_default_program()
    print("\n--- Default Program ---")
    print(json.dumps(default_prog.model_dump(), indent=2))

    print(f"\nAvailable domains for heuristics: {generator.domain_keys}")
    print(f"Available prefix keys: {generator.prefix_keys}")

    for i in range(3): # Generate a few random programs
        random_prog = generator.generate_random_program()
        print(f"\n--- Random Program {i+1} ---")
        print(json.dumps(random_prog.model_dump(), indent=2))

        mutated_random_prog = generator.mutate_program(random_prog, mutation_probability=0.5, mutation_strength=0.5)
        print(f"\n--- Mutated Random Program {i+1} (50% field mutation rate, 50% strength) ---")
        print(json.dumps(mutated_random_prog.model_dump(), indent=2))
