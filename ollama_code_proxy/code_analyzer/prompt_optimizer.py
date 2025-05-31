import random
import copy
import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple, Coroutine

from .prompt_program import PromptProgram, PromptProgramGenerator
from .evaluator import CodeEvaluator
from .context_retriever import ContextRetriever

OllamaGenerateCallable = Callable[[str, str, bool, Optional[Dict[str, Any]]], Coroutine[Any, Any, str]]

class EvaluatedProgram:
    def __init__(self, program: PromptProgram, generated_code: str, evaluation_details: Dict[str, Any], execution_time: float, full_prompt_sent: Optional[str] = None):
        self.program = program
        self.generated_code = generated_code
        self.evaluation_details = evaluation_details
        self.score = evaluation_details.get("final_score", 0.0)
        self.execution_time = execution_time
        self.full_prompt_sent = full_prompt_sent

    def __repr__(self):
        program_details = f"prefix_key='{self.program.instructional_prefix_key}', max_sym={self.program.max_primary_symbols}"
        return f"EvaluatedProgram(score={self.score:.3f}, {program_details})"

    def to_log_dict(self, user_request: str) -> Dict[str, Any]:
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()),
            "user_request_summary": user_request[:200] + ("..." if len(user_request) > 200 else ""),
            "prompt_program": self.program.model_dump(),
            # "full_prompt_sent": self.full_prompt_sent, # Logged separately or if debug
            "generated_code_summary": self.generated_code[:300] + ("..." if len(self.generated_code) > 300 else ""),
            "evaluation_details": self.evaluation_details,
            "final_score": self.score,
            "execution_time_seconds": round(self.execution_time, 3)
        }

class PrintLogger:
    def info(self, message: str): print(f"INFO: {message}")
    def error(self, message: str, exc_info:bool=False):
        print(f"ERROR: {message}")
        if exc_info and isinstance(exc_info, BaseException): # Check against BaseException
            import traceback
            traceback.print_exc() # Print full traceback
    def warning(self, message: str): print(f"WARNING: {message}")

    def log_json_record(self, record_dict: Dict[str, Any]):
        try:
            print(f"OPTIMIZER_FEEDBACK: {json.dumps(record_dict)}")
        except TypeError:
             print(f"OPTIMIZER_FEEDBACK_SERIALIZATION_ERROR: Request: {record_dict.get('user_request_summary')}, Score: {record_dict.get('final_score')}")

class PromptOptimizer:
    def __init__(self,
                 generator: PromptProgramGenerator,
                 evaluator: CodeEvaluator,
                 context_retriever: ContextRetriever,
                 ollama_generate_func: OllamaGenerateCallable,
                 default_ollama_model: str,
                 logger: Optional[Any] = None
                ):
        self.generator = generator
        self.evaluator = evaluator
        self.context_retriever = context_retriever
        self.ollama_generate_func = ollama_generate_func
        self.default_ollama_model = default_ollama_model
        self.logger = logger if logger else PrintLogger()

        self.feedback_logging_method: Callable[[Dict[str, Any]], None]
        if hasattr(self.logger, 'info') and not isinstance(self.logger, PrintLogger) and not hasattr(self.logger, 'log_json_record'):
            # Standard logger from logging module (used by main.py if OPTIMIZER_LOG_PATH is set)
            # The file handler in main.py for feedback_logger uses a simple formatter expecting a string.
            self.feedback_logging_method = lambda record_dict: self.logger.info(json.dumps(record_dict)) # type: ignore
        elif hasattr(self.logger, 'log_json_record'): # For PrintLogger or custom loggers with this method
            self.feedback_logging_method = self.logger.log_json_record
        else: # Fallback if logger has neither (should not happen with PrintLogger default)
            self.feedback_logging_method = lambda record_dict: print(str(record_dict))


    async def _execute_prompt_program(
        self, program: PromptProgram, user_request: str, current_filepath_abs: Optional[str] = None
    ) -> EvaluatedProgram:
        start_time = time.time()

        # Note: ContextRetriever's get_context_for_prompt uses its own default for max_context_tokens.
        # A deeper integration would involve PromptProgram directly parameterizing ContextRetriever's behavior.
        retrieved_context_str = self.context_retriever.get_context_for_prompt(
            user_prompt=user_request, current_filepath_abs=current_filepath_abs
        )

        instructional_prefix = self.generator.predefined_instructional_prefixes.get(
            program.instructional_prefix_key, "You are a helpful assistant."
        )

        final_prompt = f"{instructional_prefix}\n\n"
        if retrieved_context_str and "No specific code context found" not in retrieved_context_str:
            final_prompt += f"### Code Context (from codebase analysis):\n{retrieved_context_str}\n\n"
        final_prompt += f"### User Request:\n{user_request}"

        generated_code = ""
        evaluation_details: Dict[str, Any] = {"final_score": -1.0, "error": "Initialization error"}

        try:
            generated_code = await self.ollama_generate_func(
                prompt=final_prompt, model=self.default_ollama_model,
                stream=False, options={}
            )
            evaluation_details = self.evaluator.evaluate_code(generated_code)
        except Exception as e_gen_eval:
            self.logger.error(f"Exception during LLM call or evaluation for program (prefix: {program.instructional_prefix_key}): {e_gen_eval}", exc_info=e_gen_eval) # Pass exception for traceback
            evaluation_details = {"final_score": -1.0, "error": f"Execution/Eval Error: {str(e_gen_eval)}", "parsability_score": 0.0}

        end_time = time.time()
        execution_time = end_time - start_time

        return EvaluatedProgram(program, generated_code, evaluation_details, execution_time, full_prompt_sent=final_prompt)

    async def find_best_prompt_and_response(
        self, user_request: str, current_filepath_abs: Optional[str] = None,
        population_size: int = 5, num_generations: int = 1,
        elitism_count: int = 1, mutation_probability: float = 0.2,
        mutation_strength: float = 0.1
    ) -> Optional[EvaluatedProgram]:

        if population_size <= 0 or num_generations <= 0:
            self.logger.error("Population size and generations must be positive.") # type: ignore
            return None
        elitism_count = max(0, min(elitism_count, population_size // 2 if population_size > 1 else 0))

        self.logger.info(f"Starting prompt optimization for request: '{user_request[:50]}...' Pop: {population_size}, Gens: {num_generations}") # type: ignore

        population: List[PromptProgram] = [self.generator.generate_default_program()]
        for _ in range(max(0, population_size -1)):
            population.append(self.generator.generate_random_program())
        population = population[:population_size]

        overall_best_eval: Optional[EvaluatedProgram] = None

        for gen in range(num_generations):
            self.logger.info(f"--- Optimizer Generation {gen + 1}/{num_generations} ---") # type: ignore

            eval_tasks = [self._execute_prompt_program(prog, user_request, current_filepath_abs) for prog in population]
            evaluated_results_with_potential_errors = await asyncio.gather(*eval_tasks, return_exceptions=True)

            evaluated_population: List[EvaluatedProgram] = []
            for i, res_or_err in enumerate(evaluated_results_with_potential_errors):
                prog_for_log = population[i]
                if isinstance(res_or_err, EvaluatedProgram):
                    evaluated_population.append(res_or_err)
                    self.logger.info(f"  Evaluated program (Prefix: {res_or_err.program.instructional_prefix_key}) - Score: {res_or_err.score:.3f}, Time: {res_or_err.execution_time:.2f}s") # type: ignore
                    if self.feedback_logging_method: self.feedback_logging_method(res_or_err.to_log_dict(user_request))
                else:
                    self.logger.error(f"  Unexpected error evaluating program (Prefix: {prog_for_log.instructional_prefix_key}): {res_or_err}", exc_info=isinstance(res_or_err, BaseException)) # type: ignore
                    dummy_eval = EvaluatedProgram(prog_for_log, "", {"final_score": -1.0, "error": str(res_or_err)}, 0, full_prompt_sent="<error_before_prompt_gen>")
                    evaluated_population.append(dummy_eval)
                    if self.feedback_logging_method: self.feedback_logging_method(dummy_eval.to_log_dict(user_request))

            if not evaluated_population:
                self.logger.warning("No programs were successfully evaluated in this generation.") # type: ignore
                continue

            evaluated_population.sort(key=lambda x: x.score, reverse=True)
            current_gen_best = evaluated_population[0]
            self.logger.info(f"  Generation {gen+1} best score: {current_gen_best.score:.3f} (Prefix: {current_gen_best.program.instructional_prefix_key})") # type: ignore

            if overall_best_eval is None or current_gen_best.score > overall_best_eval.score:
                 overall_best_eval = current_gen_best

            if gen < num_generations - 1:
                next_population: List[PromptProgram] = []
                for i in range(min(elitism_count, len(evaluated_population))):
                    next_population.append(copy.deepcopy(evaluated_population[i].program))

                num_to_generate = population_size - len(next_population)
                for i in range(num_to_generate):
                    parent_idx = random.randint(0, max(0, (len(evaluated_population) // 2) -1)) if len(evaluated_population) > 1 else 0
                    parent_program = evaluated_population[parent_idx].program
                    mutated_child = self.generator.mutate_program(
                        parent_program, mutation_probability=mutation_probability, mutation_strength=mutation_strength
                    )
                    next_population.append(mutated_child)
                population = next_population[:population_size]

        self.logger.info(f"Optimization finished. Overall best score: {overall_best_eval.score:.3f if overall_best_eval else 'N/A'}") # type: ignore
        return overall_best_eval

if __name__ == '__main__':
    import json
    from unittest.mock import MagicMock

    print("PromptOptimizer class with logging integration defined.")

    async def mock_ollama_generate_for_opt_test(prompt: str, model: str, stream: bool, options: Optional[Dict]) -> str:
        await asyncio.sleep(0.01)
        if len(prompt) > 500 and "good" in prompt: return "def good_func():\n  '''Good doc'''\n  pass"
        elif "bad" in prompt: return "def bad_func() return 1"
        return "def ok_func(): pass"

    async def main_optimizer_example_with_logging():
        print("\n--- PromptOptimizer Example with Logging ---")

        generator = PromptProgramGenerator()
        evaluator = CodeEvaluator()

        mock_context_retriever = MagicMock(spec=ContextRetriever)
        mock_context_retriever.get_context_for_prompt.return_value = (
            "### Context from `dummy.py`:\n```python\ndef existing_function(): pass\n```"
        )
        mock_context_retriever.config = {"max_context_tokens": 1500} # Mock attribute if accessed

        example_logger = PrintLogger()

        optimizer = PromptOptimizer(
            generator=generator, evaluator=evaluator, context_retriever=mock_context_retriever,
            ollama_generate_func=mock_ollama_generate_for_opt_test,
            default_ollama_model="test-model", logger=example_logger
        )

        user_request = "Write a Python function to calculate factorial, make it good."

        print(f"Optimizing for request: '{user_request}'")
        best_result = await optimizer.find_best_prompt_and_response(
            user_request=user_request, population_size=3, num_generations=2, elitism_count=1
        )

        if best_result:
            print("\n--- Best Result Found (Optimizer Example) ---")
            print(f"Score: {best_result.score:.3f}")
            print("Prompt Program Parameters (example):")
            print(json.dumps(best_result.program.model_dump(exclude={'domain_heuristic_strengths'}), indent=2))
            print("\nGenerated Code (example):")
            print(best_result.generated_code)
        else:
            print("No result from optimizer example.")

    # To run this example:
    # asyncio.run(main_optimizer_example_with_logging())
    # print("\nNote: To run the async main_optimizer_example_with_logging, uncomment asyncio.run(...) above.")
    # print("This script primarily defines the classes; direct execution of example is for dev testing.")
