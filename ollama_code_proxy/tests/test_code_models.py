import unittest
# Correct the import path based on project structure
# Assuming tests are run from the 'ollama_code_proxy' root directory
from code_analyzer.models import (
    ImportInfo,
    ArgumentInfo,
    DecoratorInfo,
    FunctionInfo,
    MethodInfo,
    ClassInfo,
    VariableInfo,
    ModuleInfo
)

class TestCodeModels(unittest.TestCase):

    def test_module_info_instantiation(self):
        module = ModuleInfo(module_name="test_mod", filepath="test_mod.py")
        self.assertEqual(module.module_name, "test_mod")
        self.assertEqual(module.filepath, "test_mod.py")
        self.assertEqual(module.imports, [])
        self.assertEqual(module.functions, [])
        self.assertEqual(module.classes, [])
        self.assertEqual(module.variables, [])

    def test_function_info_with_all_fields(self):
        arg = ArgumentInfo(name="param1", annotation="int", default_value="100")
        decorator = DecoratorInfo(name="@my_decorator", lineno=1)
        func = FunctionInfo(
            name="sample_func",
            lineno=2,
            end_lineno=5,
            docstring="A sample function.",
            args=[arg],
            returns="str",
            decorators=[decorator]
        )
        self.assertEqual(func.name, "sample_func")
        self.assertEqual(func.args[0].name, "param1")
        self.assertEqual(func.decorators[0].name, "@my_decorator")

    def test_class_info_with_methods_and_variables(self):
        method_arg = ArgumentInfo(name="self")
        method = MethodInfo(
            name="do_something",
            lineno=5,
            args=[method_arg],
            returns="None"
        )
        class_var = VariableInfo(
            name="MAX_SIZE",
            lineno=3,
            annotation="int"
        )
        class_info = ClassInfo(
            name="MyTestClass",
            lineno=2,
            end_lineno=10,
            bases=["BaseObject"],
            methods=[method],
            class_variables=[class_var],
            decorators=[]
        )
        self.assertEqual(class_info.name, "MyTestClass")
        self.assertEqual(len(class_info.methods), 1)
        self.assertEqual(class_info.methods[0].name, "do_something")
        self.assertEqual(len(class_info.class_variables), 1)
        self.assertEqual(class_info.class_variables[0].name, "MAX_SIZE")
        self.assertListEqual(class_info.bases, ["BaseObject"])

    def test_import_info_variations(self):
        imp1 = ImportInfo(name="os", lineno=1)
        imp2 = ImportInfo(module="math", name="sqrt", asname="square_root", lineno=2)
        imp3 = ImportInfo(module=".local_mod", name="LocalUtil", level=1, lineno=3)

        self.assertEqual(imp1.name, "os")
        self.assertEqual(imp2.module, "math")
        self.assertEqual(imp3.level, 1)

if __name__ == '__main__':
    # This allows running the tests directly from the file,
    # but we'll primarily rely on pytest or `python -m unittest discover`
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
