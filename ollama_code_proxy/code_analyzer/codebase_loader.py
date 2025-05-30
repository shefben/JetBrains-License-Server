import os
from pathlib import Path
from typing import List, Dict, Optional

from .parser import PythonParser # Assuming parser.py is in the same directory
from .models import ModuleInfo   # Assuming models.py is in the same directory

# Supported languages and their default extensions
SUPPORTED_LANGUAGES = {
    "python": [".py"]
}

class CodebaseLoader:
    """
    Discovers, loads, and parses code files from a specified directory.
    It uses language-specific parsers to analyze the files and stores
    the structured information.
    """
    def __init__(self,
                 supported_languages: Optional[Dict[str, List[str]]] = None,
                 progress_callback: Optional[callable] = None):
        """
        Initializes the CodebaseLoader.

        Args:
            supported_languages: A dictionary mapping language names to lists of file extensions.
                                 Defaults to Python (.py) files.
            progress_callback: An optional function that can be called to report progress
                               (e.g., progress_callback(filepath, index, total_files)).
        """
        self.supported_languages = supported_languages or SUPPORTED_LANGUAGES
        self.progress_callback = progress_callback
        self.parsed_modules: List[ModuleInfo] = []
        # In a more advanced scenario, parsers could be dynamically loaded based on language.
        # For now, we'll hardcode PythonParser for simplicity as it's the only one implemented.
        self.parsers = {
            "python": PythonParser()
        }

    def load_from_directory(self, directory_path: str, excluded_dirs: Optional[List[str]] = None, excluded_files: Optional[List[str]] = None) -> None:
        """
        Loads and parses all supported code files from the given directory and its subdirectories.

        Args:
            directory_path: The root directory to start scanning for code files.
            excluded_dirs: A list of directory names to exclude (e.g., ['.git', 'venv', '__pycache__']).
                           Defaults to common exclusions.
            excluded_files: A list of file names or patterns to exclude.
        """
        self.parsed_modules = [] # Reset before loading

        if excluded_dirs is None:
            excluded_dirs = ['.git', 'venv', '__pycache__', 'node_modules', '.vscode', '.idea']
        if excluded_files is None:
            excluded_files = [] # Add any specific files like 'setup.py' if needed by default

        root_path = Path(directory_path)
        if not root_path.is_dir():
            raise ValueError(f"Provided path '{directory_path}' is not a valid directory.")

        # First pass: collect all files to process for accurate progress reporting
        files_to_process = []
        for lang_name, extensions in self.supported_languages.items():
            if lang_name not in self.parsers:
                # print(f"Warning: No parser available for language '{lang_name}'. Skipping.")
                continue
            for ext in extensions:
                for filepath in root_path.rglob(f"*{ext}"):
                    if self._is_excluded(filepath, root_path, excluded_dirs, excluded_files):
                        continue
                    files_to_process.append(filepath)

        total_files = len(files_to_process)
        for i, filepath in enumerate(files_to_process):
            lang_name = self._get_language_from_extension(filepath.suffix)
            if not lang_name: # Should not happen if collected correctly
                continue

            if self.progress_callback:
                try:
                    self.progress_callback(str(filepath), i + 1, total_files)
                except Exception as e: # Don't let callback errors stop processing
                    print(f"Error in progress_callback for {filepath}: {e}")

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()

                parser = self.parsers[lang_name]
                parsed_data = parser.parse(code) # This returns a Dict

                # Construct ModuleInfo object
                module_info = ModuleInfo(
                    module_name=filepath.stem, # Use filename without extension as module name
                    filepath=str(filepath.relative_to(root_path)), # Store relative path
                    language=lang_name, # Add language to ModuleInfo if it's part of the model
                    docstring=parsed_data.get("docstring"), # Assuming parser extracts module docstring
                    imports=parsed_data.get("imports", []),
                    functions=parsed_data.get("functions", []),
                    classes=parsed_data.get("classes", []),
                    variables=parsed_data.get("variables", []),
                    parse_errors= [parsed_data["error"]] if "error" in parsed_data else []
                )
                # Note: The PythonParser currently returns a dict where errors are at the top level.
                # The ModuleInfo model expects parse_errors as a list.
                # We need to adapt this if the parser directly returns ModuleInfo or if ModuleInfo's
                # fields directly match the parser's output structure for elements like functions, classes etc.
                # For now, we are mapping fields. This also means PythonParser needs to be updated
                # to provide module-level docstring and a clearer error structure if ModuleInfo is to be
                # built this way.
                # Let's assume for now the parser's output dict has keys that match ModuleInfo fields.

                self.parsed_modules.append(module_info)

            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
                # Optionally, create a ModuleInfo with error information
                error_module = ModuleInfo(
                    module_name=filepath.stem,
                    filepath=str(filepath.relative_to(root_path)),
                    language=lang_name,
                    parse_errors=[{"error": f"Failed to read or parse file: {str(e)}", "details": str(e)}]
                )
                self.parsed_modules.append(error_module)


    def _is_excluded(self, filepath: Path, root_path: Path, excluded_dirs: List[str], excluded_files: List[str]) -> bool:
        """Checks if a file or its parent directories are in the exclusion lists."""
        # Check against excluded directory names anywhere in the path
        try:
            relative_path_parts = filepath.relative_to(root_path).parts
        except ValueError: # filepath is not under root_path, should not happen with rglob
            return True

        for part in relative_path_parts[:-1]: # Check directory parts
            if part in excluded_dirs:
                return True

        # Check against excluded file names
        if filepath.name in excluded_files:
            return True

        # Add more sophisticated pattern matching for excluded_files if needed (e.g., using fnmatch)
        return False

    def _get_language_from_extension(self, extension: str) -> Optional[str]:
        """Gets the language name from a file extension."""
        for lang, exts in self.supported_languages.items():
            if extension.lower() in exts:
                return lang
        return None

    def get_all_modules(self) -> List[ModuleInfo]:
        """Returns all parsed ModuleInfo objects."""
        return self.parsed_modules

    def get_module_by_filepath(self, relative_filepath: str) -> Optional[ModuleInfo]:
        """Retrieves a parsed module by its relative filepath."""
        for module in self.parsed_modules:
            if module.filepath == relative_filepath:
                return module
        return None

if __name__ == '__main__':
    # Example Usage (for demonstration)

    # Define a simple progress callback
    def my_progress_reporter(filepath, current, total):
        print(f"Processing ({current}/{total}): {filepath}")

    # Create a loader instance
    loader = CodebaseLoader(progress_callback=my_progress_reporter)

    # Create dummy files and directory structure for testing
    # In a real scenario, this path would be an existing codebase.
    dummy_project_path = Path("_temp_dummy_project")

    if not dummy_project_path.exists():
        dummy_project_path.mkdir()
        (dummy_project_path / "module1.py").write_text(
            "import os\\n\\ndef func1():\\n    pass\\n\\nclass ClassA:\\n    def method_a(self):\\n        return os.name"
        )
        (dummy_project_path / "module2.py").write_text(
            "def func2(x, y):\\n    return x + y\\n\\nMY_VAR = 100"
        )
        sub_dir = dummy_project_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "module3.py").write_text(
            "from ..module1 import func1 # Relative import\\n\\ndef func3():\\n    func1()"
        )
        # Add an excluded directory and file
        excluded_dir_path = dummy_project_path / "venv"
        excluded_dir_path.mkdir()
        (excluded_dir_path / "some_venv_file.py").write_text("pass")
        (dummy_project_path / ".env").write_text("SECRET=123") # Example of an excluded file if we add it

    print(f"Loading from: {dummy_project_path.resolve()}")
    try:
        loader.load_from_directory(str(dummy_project_path), excluded_files=['.env'])

        all_modules = loader.get_all_modules()
        print(f"\\nFound and parsed {len(all_modules)} modules:")
        for mod_info in all_modules:
            print(f"  Module: {mod_info.module_name} ({mod_info.filepath})")
            if mod_info.functions:
                print(f"    Functions: {', '.join([f.name for f in mod_info.functions])}")
            if mod_info.classes:
                print(f"    Classes: {', '.join([c.name for c in mod_info.classes])}")
            if mod_info.variables:
                print(f"    Variables: {', '.join([v.name for v in mod_info.variables])}")
            if mod_info.parse_errors:
                print(f"    Parse Errors: {mod_info.parse_errors}")

        mod1_path = "module1.py" # Relative path
        retrieved_mod1 = loader.get_module_by_filepath(mod1_path)
        if retrieved_mod1:
            print(f"\\nRetrieved module by path '{mod1_path}': {retrieved_mod1.module_name}")
        else:
            print(f"\\nCould not retrieve module by path '{mod1_path}'")

    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up dummy files (optional, good for testing)
        import shutil
        if dummy_project_path.exists():
             shutil.rmtree(dummy_project_path)
        pass
