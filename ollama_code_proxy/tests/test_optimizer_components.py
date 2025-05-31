import pytest
import os
import asyncio
from typing import Dict, List, Optional, Any, Coroutine, Callable, Tuple # Added Tuple
from unittest.mock import MagicMock, AsyncMock, patch

from ollama_code_proxy.code_analyzer.prompt_program import PromptProgram, PromptProgramGenerator
from ollama_code_proxy.code_analyzer.evaluator import CodeEvaluator
from ollama_code_proxy.code_analyzer.context_retriever import ContextRetriever
from ollama_code_proxy.code_analyzer.prompt_optimizer import PromptOptimizer, EvaluatedProgram, OllamaGenerateCallable, PrintLogger
from ollama_code_proxy.code_analyzer.knowledge_graph import KnowledgeGraph
from ollama_code_proxy.code_analyzer.models import ModuleInfo
from fastapi import Response as FastAPIResponse # For mocking a specific return type if needed by a test

# --- Tests for PromptProgram ---
def test_prompt_program_defaults():
    prog = PromptProgram()
    assert prog.max_primary_symbols == 2
    assert prog.include_callees is True
    assert prog.instructional_prefix_key == "default_v1"
    assert prog.domain_heuristic_strengths.get("database") == 0.5
    assert prog.domain_heuristic_strengths.get("test") == 0.7

def test_prompt_program_validation():
    with pytest.raises(ValueError):
        PromptProgram(max_primary_symbols=0)
    with pytest.raises(ValueError):
        PromptProgram(callee_snippet_max_lines=101) # le=30 was in script, but model has 100 for primary_symbol_snippet_max_lines, let's assume similar for others or check model defs
                                                 # Current model has callee_snippet_max_lines le=30. So 100 is out of bounds.
    prog = PromptProgram(max_primary_symbols=3)
    assert prog.max_primary_symbols == 3

    new_strengths = {"database": 0.9, "new_domain_ignored": 0.4}
    prog.domain_heuristic_strengths = new_strengths
    assert prog.domain_heuristic_strengths["database"] == 0.9
    assert "new_domain_ignored" not in prog.domain_heuristic_strengths


# --- Tests for PromptProgramGenerator ---
@pytest.fixture
def prog_generator() -> PromptProgramGenerator:
    custom_prefixes = {
        "custom_v1": "Custom prefix.",
        "another_custom": "Another one."
    }
    return PromptProgramGenerator(predefined_instructional_prefixes=custom_prefixes)

def test_ppg_generate_default(prog_generator: PromptProgramGenerator):
    prog = prog_generator.generate_default_program()
    assert prog.instructional_prefix_key == "default_v1"

    default_gen_no_custom_prefix = PromptProgramGenerator()
    prog_default_prefix = default_gen_no_custom_prefix.generate_default_program()
    assert prog_default_prefix.instructional_prefix_key == "default_v1"
    assert "default_v1" in default_gen_no_custom_prefix.predefined_instructional_prefixes
    assert "custom_v1" not in default_gen_no_custom_prefix.prefix_keys

def test_ppg_generate_random(prog_generator: PromptProgramGenerator):
    progs = [prog_generator.generate_random_program() for _ in range(10)]
    prefix_keys_used = {p.instructional_prefix_key for p in progs}
    assert "custom_v1" in prog_generator.prefix_keys
    assert any(key in prog_generator.prefix_keys for key in prefix_keys_used)

    for p in progs:
        # Check constraints from PromptProgram model by trying to instantiate with the params
        try:
            PromptProgram(**p.model_dump())
        except ValueError as e:
            pytest.fail(f"Randomly generated program failed validation: {p.model_dump()} \nError: {e}")

        # Specific checks based on known default constraints
        assert 1 <= p.max_primary_symbols <= 5
        for strength_val in p.domain_heuristic_strengths.values():
            assert 0.0 <= strength_val <= 1.0

def test_ppg_mutate_program(prog_generator: PromptProgramGenerator):
    original_prog = prog_generator.generate_default_program()
    fields_mutated_flags = {field_name: False for field_name in PromptProgram.model_fields}

    for i in range(100): # Increase trials to raise chance of mutating all mutable fields
        mutated_prog = prog_generator.mutate_program(original_prog, mutation_probability=0.3, mutation_strength=0.5)
        for field_name in PromptProgram.model_fields:
            if getattr(mutated_prog, field_name) != getattr(original_prog, field_name):
                fields_mutated_flags[field_name] = True
        try:
            PromptProgram(**mutated_prog.model_dump()) # Validate mutated program
        except ValueError as e:
            pytest.fail(f"Mutated program failed validation: {mutated_prog.model_dump()} \nError: {e}")

    # Check that a significant number of fields were mutated at least once
    # Not all fields might mutate if mutation brings it back to original or if it's not selected by probability.
    # print(f"DEBUG: Mutation flags: {fields_mutated_flags}")
    assert sum(fields_mutated_flags.values()) >= len(PromptProgram.model_fields) // 2


# --- Tests for CodeEvaluator ---
@pytest.fixture
def code_evaluator() -> CodeEvaluator:
    return CodeEvaluator()

def test_code_evaluator_parsability(code_evaluator: CodeEvaluator):
    parsable_code = "def foo(): return 1"
    unparsable_code = "def bar() return 0"

    eval_parsable = code_evaluator.evaluate_code(parsable_code)
    assert eval_parsable["parsability_score"] == 1.0
    assert eval_parsable["final_score"] > 0.1

    eval_unparsable = code_evaluator.evaluate_code(unparsable_code)
    assert eval_unparsable["parsability_score"] == 0.0
    assert eval_unparsable["final_score"] <= 0.1 * CodeEvaluator().weights["parsability"]

def test_code_evaluator_docstrings(code_evaluator: CodeEvaluator):
    code_with_docs = "def foo():\n  '''doc'''\n  pass"
    code_without_docs = "def foo():\n  pass"
    assert code_evaluator.evaluate_code(code_with_docs)["docstring_score"] > code_evaluator.evaluate_code(code_without_docs)["docstring_score"]

# --- Tests for PromptOptimizer ---
@pytest.fixture
def mock_dependencies_for_optimizer(tmp_path, prog_generator: PromptProgramGenerator) -> Dict[str, Any]:
    evaluator = CodeEvaluator()
    mock_cr = MagicMock(spec=ContextRetriever)
    # ContextRetriever needs codebase_root. It's passed to __init__.
    # The get_context_for_prompt method also needs it implicitly if using self.codebase_root.
    # So, ensure the mock has it if any method called on it would need it.
    # For this test, ContextRetriever is directly mocked, so its internal codebase_root is not used by the mock's return value.
    mock_cr.get_context_for_prompt.return_value = "### Mocked Context:\ndef sample(): pass"
    # If ContextRetriever's get_context_for_prompt accesses self.config for max_context_tokens
    mock_cr.config = {"max_context_tokens": 1500}


    mock_ollama_gen_func = AsyncMock(spec=OllamaGenerateCallable)

    return {
        "generator": prog_generator,
        "evaluator": evaluator,
        "context_retriever": mock_cr,
        "ollama_generate_func": mock_ollama_gen_func,
        "default_ollama_model": "test-optimizer-model",
        "logger": PrintLogger() # Use PrintLogger for test output visibility
    }

@pytest.mark.asyncio
async def test_prompt_optimizer_execute_program(mock_dependencies_for_optimizer: Dict[str, Any]):
    optimizer = PromptOptimizer(**mock_dependencies_for_optimizer)
    program = optimizer.generator.generate_default_program()
    user_request = "Test request for code generation"

    optimizer.ollama_generate_func.return_value = "def generated_func_example():\n    return 'optimized'" # type: ignore

    evaluated_prog = await optimizer._execute_prompt_program(program, user_request)

    assert isinstance(evaluated_prog, EvaluatedProgram)
    assert evaluated_prog.program == program
    assert "generated_func_example" in evaluated_prog.generated_code
    assert "final_score" in evaluated_prog.evaluation_details
    assert evaluated_prog.score >= 0.0
    optimizer.ollama_generate_func.assert_called_once() # type: ignore

    call_args_list = optimizer.ollama_generate_func.call_args_list # type: ignore
    assert len(call_args_list) == 1
    args_tuple, kwargs_dict = call_args_list[0] # call_args is a tuple (args, kwargs)
    sent_prompt = kwargs_dict.get('prompt', args_tuple[0] if args_tuple else '')

    assert user_request in sent_prompt
    assert "Mocked Context" in sent_prompt
    assert program.instructional_prefix_key in sent_prompt

@pytest.mark.asyncio
async def test_prompt_optimizer_find_best_program(mock_dependencies_for_optimizer: Dict[str, Any]):
    optimizer = PromptOptimizer(**mock_dependencies_for_optimizer)
    user_request = "Find the best program to generate a good function."

    async def conditional_response_mock(prompt: str, model: str, stream: bool, options: Optional[Dict]):
        if "completion_v1" in prompt:
            return "def high_score_func():\n  '''This function has a good docstring and reasonable length.'''\n  return 100"
        elif "debug_v1" in prompt:
            return "def error_prone_func() return 'syntax error here'"
        return "def medium_score_func():\n  pass # Mediocre code"

    optimizer.ollama_generate_func.side_effect = conditional_response_mock # type: ignore

    # Ensure generator can produce programs with these specific keys
    custom_prefixes_for_test = {
        "completion_v1": "Completion prefix",
        "debug_v1": "Debug prefix",
        "default_v1": optimizer.generator.predefined_instructional_prefixes["default_v1"] # Keep default
    }
    optimizer.generator = PromptProgramGenerator(predefined_instructional_prefixes=custom_prefixes_for_test)


    best_eval = await optimizer.find_best_prompt_and_response(
        user_request=user_request, population_size=3, num_generations=2, elitism_count=1
    )

    assert best_eval is not None
    assert "high_score_func" in best_eval.generated_code
    assert best_eval.program.instructional_prefix_key == "completion_v1"
    assert best_eval.score > 0.5 # Assuming high_score_func scores well based on evaluator
    # Check that the mocked ollama_generate_func was called multiple times (pop_size * num_gens)
    assert optimizer.ollama_generate_func.call_count == 3 * 2 # type: ignore
