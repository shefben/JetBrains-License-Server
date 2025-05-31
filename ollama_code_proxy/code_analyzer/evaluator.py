import ast
import re
from typing import Dict, List, Optional, Any, Tuple

# Radon is not available in this environment, so keep this flag False.
RADON_AVAILABLE = False

class CodeEvaluator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.weights = {
            "parsability": self.config.get("weight_parsability", 1.0),
            "length": self.config.get("weight_length", 0.2),
            "docstrings": self.config.get("weight_docstrings", 0.3),
            "anti_patterns": self.config.get("weight_anti_patterns", 0.3),
            "placeholders": self.config.get("weight_placeholders", 0.2),
            # "complexity": self.config.get("weight_complexity", 0.1), # If Radon was used
        }

        self.optimal_min_len_chars = self.config.get("optimal_min_len_chars", 50)
        self.optimal_max_len_chars = self.config.get("optimal_max_len_chars", 2500)
        self.length_penalty_factor = self.config.get("length_penalty_factor", 0.5)


    def _check_parsability(self, code_string: str) -> Tuple[float, Optional[ast.AST]]:
        try:
            tree = ast.parse(code_string)
            return 1.0, tree
        except SyntaxError:
            return 0.0, None
        except Exception:
            return 0.0, None

    def _score_code_length(self, code_string: str) -> float:
        length = len(code_string)
        if self.optimal_min_len_chars <= length <= self.optimal_max_len_chars:
            return 1.0

        if length < self.optimal_min_len_chars:
            if self.optimal_min_len_chars == 0: return self.length_penalty_factor
            score = self.length_penalty_factor + (1.0 - self.length_penalty_factor) * (length / self.optimal_min_len_chars)
            return max(0.0, score)

        # length > self.optimal_max_len_chars
        penalty_range = self.optimal_max_len_chars
        if penalty_range == 0: return self.length_penalty_factor

        over_length = length - self.optimal_max_len_chars
        score_reduction = (1.0 - self.length_penalty_factor) * min(over_length / penalty_range, 1.0)
        score = 1.0 - score_reduction
        return max(0.0, score)

    def _score_docstring_presence(self, tree: ast.AST) -> float:
        total_definable_nodes = 0
        nodes_with_docstrings = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                total_definable_nodes += 1
                if ast.get_docstring(node, clean=False):
                    nodes_with_docstrings += 1
                if isinstance(node, ast.ClassDef):
                    for sub_node in node.body:
                        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                             total_definable_nodes += 1
                             if ast.get_docstring(sub_node, clean=False):
                                 nodes_with_docstrings += 1
        if total_definable_nodes == 0: return 0.5
        return nodes_with_docstrings / total_definable_nodes

    def _score_anti_patterns(self, code_string: str) -> float:
        score = 1.0
        if re.search(r"\bTODO\b", code_string, re.IGNORECASE): score -= 0.2
        if re.search(r"\bFIXME\b", code_string, re.IGNORECASE): score -= 0.2
        if "NotImplementedError" in code_string: score -= 0.3
        standalone_pass_matches = re.findall(r"^\s*pass\s*$", code_string, re.MULTILINE)
        if len(standalone_pass_matches) > 0:
            score -= min(0.3, 0.1 * len(standalone_pass_matches)) # Cap penalty for many passes
        return max(0.0, score)

    def _score_placeholder_usage(self, code_string: str) -> float:
        score = 1.0
        placeholders_patterns = [
            r"\[\s*insert code here\s*\]", r"\[\s*your code here\s*\]",
            r"#\s*TODO:\s*Implement", r"#\s*implement logic here",
            r"#\s*add your code here"
        ]
        if re.search(r"^\s*\.\.\.\s*$", code_string, re.MULTILINE): score -= 0.4
        elif "..." in code_string and re.search(r"(?<!\w)\.\.\.(?!\w)", code_string): # Ellipsis not part of other syntax
             score -= 0.15
        for p_holder_pattern in placeholders_patterns:
            if re.search(p_holder_pattern, code_string, re.IGNORECASE): score -= 0.3
        return max(0.0, score)

    def evaluate_code(self, code_string: str) -> Dict[str, Any]:
        results: Dict[str, Any] = { # Ensure type consistency for values
            "final_score": 0.0, "parsability_score": 0.0, "length_score": 0.0,
            "docstring_score": 0.0, "anti_pattern_score": 1.0, "placeholder_score": 1.0,
            "error": None
        }
        if not code_string or code_string.isspace():
            results["error"] = "Code string is empty or whitespace."
            return results

        parsability_score, tree = self._check_parsability(code_string)
        results["parsability_score"] = parsability_score

        if parsability_score == 0.0:
            results["error"] = "Code does not parse."
            results["length_score"] = self._score_code_length(code_string) # Still score length
            results["final_score"] = 0.1 * self.weights["parsability"] * (0.5 + 0.5 * results["length_score"])
            return results

        results["length_score"] = self._score_code_length(code_string)
        results["docstring_score"] = self._score_docstring_presence(tree) if tree else 0.0
        results["anti_pattern_score"] = self._score_anti_patterns(code_string)
        results["placeholder_score"] = self._score_placeholder_usage(code_string)

        contributing_weights_sum = (
            self.weights["length"] + self.weights["docstrings"] +
            self.weights["anti_patterns"] + self.weights["placeholders"]
        )
        if contributing_weights_sum == 0: contributing_weights_sum = 1.0

        weighted_score_sum = (
            (self.weights["length"] * results["length_score"]) +
            (self.weights["docstrings"] * results["docstring_score"]) +
            (self.weights["anti_patterns"] * results["anti_pattern_score"]) +
            (self.weights["placeholders"] * results["placeholder_score"])
        )
        normalized_score = weighted_score_sum / contributing_weights_sum
        results["final_score"] = max(0.0, min(1.0, normalized_score))
        return results

if __name__ == '__main__':
    import json
    print("CodeEvaluator class defined.")
    evaluator = CodeEvaluator()
    test_cases = {
        "OK Code": """
def greet(name: str) -> str:
    '''Greets a person.'''
    return f"Hello, {name}!"

class User:
    '''Represents a user.'''
    def __init__(self, username: str):
        self.username = username
    def get_name(self) -> str: # Method with docstring
        '''Gets username'''
        return self.username
""",
        "No Docstrings": "def foo(a,b):\n    # This function has no docstring\n    return a+b*2- (a**b) /len('test')",
        "Syntax Error": "def func(a,b\n  return a+b",
        "Too Short": "x=1",
        "Too Long": "def long_func():\n" + "    x=1 # comment to make it longer\n" * 500,
        "Anti-Patterns": "def problem():\n    # TODO: fix this later\n    pass\n    raise NotImplementedError # FIXME",
        "Placeholders": "def solution():\n    # ... implement logic here ...\n    # [insert code here]",
        "Empty Code": "",
        "Whitespace Code": "    \n  \t  ",
        "Just Ellipsis": "...",
        "Good Short Func": "def add(x,y):\n  '''Adds two numbers'''\n  return x+y"
    }
    print("\n--- Code Evaluation Examples ---")
    for name, code in test_cases.items():
        scores = evaluator.evaluate_code(code)
        print(f"\n--- {name} ---")
        print(f"Scores: {json.dumps(scores, indent=2)}")

    custom_config = {"optimal_min_len_chars": 10, "weight_docstrings": 0.8, "length_penalty_factor": 0.2}
    evaluator_custom = CodeEvaluator(config=custom_config)
    print("\n--- Custom Config Evaluation ---")
    score_no_doc_custom = evaluator_custom.evaluate_code(test_cases["No Docstrings"])
    print(f"No Docstrings (Custom): {json.dumps(score_no_doc_custom, indent=2)}")
    score_short_custom = evaluator_custom.evaluate_code(test_cases["Too Short"])
    print(f"Too Short (Custom): {json.dumps(score_short_custom, indent=2)}")
