#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

# pylint: disable=protected-access

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
    register,
    register_module,
)
from monic.expressions.registry import registry, NamespaceProxy


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    registry.reset()
    yield


def test_registered_function():
    """Test using registered functions."""

    @register
    def custom_add(x, y):  # pylint: disable=unused-variable
        """Custom addition function."""
        return x + y

    @register("multiply")
    def custom_multiply(x, y):  # pylint: disable=unused-variable
        """Custom multiplication function."""
        return x * y

    code = """
    result1 = custom_add(5, 3)
    result2 = multiply(4, 6)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result1") == 8
    assert interpreter.get_name_value("result2") == 24


def test_registered_class():
    """Test using registered class."""

    @register
    class Point:  # pylint: disable=unused-variable
        """A simple 2D point class."""

        def __init__(self, x, y):
            self.x = x
            self.y = y

        def distance_from_origin(self):
            return (self.x**2 + self.y**2) ** 0.5

    code = """
    p = Point(3, 4)
    distance = p.distance_from_origin()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("distance") == 5.0


def test_register_without_name():
    """Test registering an object without explicit name."""

    @register
    def test_func():  # pylint: disable=unused-variable
        return 42

    code = "result = test_func()"
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result") == 42


def test_register_with_name():
    """Test registering an object with explicit name."""

    @register("answer")
    def get_answer():  # pylint: disable=unused-variable
        return 42

    code = "result = answer()"
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result") == 42


def test_register_with_nested_name():
    """Test registering objects with nested names."""

    @register("math.functions.add")
    def add(x, y):  # pylint: disable=unused-variable
        return x + y

    @register("math.functions.multiply")
    def multiply(x, y):  # pylint: disable=unused-variable
        return x * y

    @register("math.constants")
    class MathConstants:  # pylint: disable=unused-variable
        PI = 3.14159
        E = 2.71828

    code = """
    result1 = math.functions.add(5, 3)
    result2 = math.functions.multiply(4, 2)
    result3 = math.constants.PI
    result4 = math.constants.E
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result1") == 8
    assert interpreter.get_name_value("result2") == 8
    assert abs(interpreter.get_name_value("result3") - 3.14159) < 1e-5
    assert abs(interpreter.get_name_value("result4") - 2.71828) < 1e-5


def test_register_nested_name_conflict():
    """Test that registering conflicting nested names raises an error."""

    @register("math.functions.add")
    def add1(x, y):  # pylint: disable=unused-variable
        return x + y  # pragma: no cover

    # Try to register another function with the same nested name
    with pytest.raises(ValueError) as exc_info:

        @register("math.functions.add")
        def add2(x, y):  # pylint: disable=unused-variable
            return x + y  # pragma: no cover

    assert "is already registered in namespace" in str(exc_info.value)


def test_register_nested_name_non_namespace_conflict():
    """Test conflict between nested name and non-namespace object."""

    @register("math")
    def math_func():  # pylint: disable=unused-variable
        return 42  # pragma: no cover

    # Try to register a function in math.functions namespace
    with pytest.raises(ValueError) as exc_info:

        @register("math.functions.add")
        def add(x, y):  # pylint: disable=unused-variable
            return x + y  # pragma: no cover

    assert "is already registered as a non-namespace" in str(exc_info.value)


def test_register_both_syntaxes():
    """Test both decorator syntaxes work correctly."""

    @register
    def func1():  # pylint: disable=unused-variable
        return 1

    @register()
    def func2():  # pylint: disable=unused-variable
        return 2

    code = """
    result1 = func1()
    result2 = func2()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result1") == 1
    assert interpreter.get_name_value("result2") == 2


def test_register_class_both_syntaxes():
    """Test both decorator syntaxes work correctly with classes."""

    @register
    class Class1:  # pylint: disable=unused-variable
        value = 1

    @register()
    class Class2:  # pylint: disable=unused-variable
        value = 2

    code = """
    obj1 = Class1()
    obj2 = Class2()
    result1 = obj1.value
    result2 = obj2.value
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result1") == 1
    assert interpreter.get_name_value("result2") == 2


def test_register_module():
    """Test registering and using a module."""
    # Register math module
    register_module("math")

    code = """
    result1 = math.sqrt(16)
    result2 = math.pi
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result1") == 4.0
    assert (
        abs(interpreter.get_name_value("result2") - 3.141592653589793) < 1e-10
    )


def test_register_module_with_alias():
    """Test registering a module with an alias."""
    # Register random module with alias
    register_module("random", alias="rand")

    code = """
    rand.seed(42)  # For reproducibility
    result = rand.randint(1, 10)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    # The actual value doesn't matter as long as it's in range
    result = interpreter.get_name_value("result")
    assert 1 <= result <= 10


def test_register_module_submodule():
    """Test registering a module with submodules."""
    # Register collections.abc module
    register_module("collections.abc", alias="collabc")

    code = """
    is_sequence = collabc.Sequence
    result = isinstance([1, 2, 3], is_sequence)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    assert result is True


def test_register_module_with_nested_name():
    """Test registering modules with nested names."""
    # Register urllib and its parse submodule
    register_module("urllib", alias="url")
    register_module("urllib.parse", alias="url.parse")

    code = """
    # Use urllib.parse module
    query = url.parse.urlencode({'name': 'test', 'value': 42})
    parsed = url.parse.urlparse('https://example.com/path?key=value')
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    query = interpreter.get_name_value("query")
    parsed = interpreter.get_name_value("parsed")

    assert query == "name=test&value=42"
    assert parsed.scheme == "https"
    assert parsed.netloc == "example.com"
    assert parsed.path == "/path"


def test_register_non_nested_name_conflict():
    """
    Test that registering a non-nested name that already exists raises an error.
    """
    namespace = {}
    registry._register_in_namespace("test", 42, namespace)

    with pytest.raises(ValueError) as exc_info:
        registry._register_in_namespace("test", 43, namespace)

    assert "Name 'test' is already registered" in str(exc_info.value)


def test_namespace_proxy_attribute_error():
    """
    Test that accessing non-existent attribute in NamespaceProxy raises
    AttributeError.
    """
    namespace = {"a": 1}
    proxy = NamespaceProxy(namespace)

    with pytest.raises(AttributeError) as exc_info:
        _ = proxy.non_existent

    assert "'non_existent' not found in namespace" in str(exc_info.value)


def test_register_object_without_name():
    """
    Test that registering an object without a name and no __name__ attribute
    raises ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        registry.register()(object())

    assert "No name provided and object has no __name__ attribute" in str(
        exc_info.value
    )


def test_register_module_import_error():
    """Test that registering a non-existent module raises ImportError."""
    with pytest.raises(ImportError) as exc_info:
        register_module("non_existent_module")

    assert "Failed to import module 'non_existent_module'" in str(
        exc_info.value
    )


def test_register_module_duplicate():
    """Test that registering the same module twice raises ValueError."""
    register_module("math")

    with pytest.raises(ValueError) as exc_info:
        register_module("math")

    assert "Module 'math' is already registered" in str(exc_info.value)


def test_get_all_with_nested_namespaces():
    """Test that get_all properly handles nested namespaces."""

    @register("math.functions.add")
    def add(x, y):  # pylint: disable=unused-variable
        return x + y  # pragma: no cover

    @register("math.constants")
    class Constants:  # pylint: disable=unused-variable
        PI = 3.14159

    all_objects = registry.get_all()
    assert isinstance(all_objects["math"], NamespaceProxy)
    assert isinstance(all_objects["math"].functions, NamespaceProxy)
    assert callable(all_objects["math"].functions.add)
    assert all_objects["math"].constants.PI == 3.14159


def test_register_callable_without_name():
    """
    Test that registering a callable without __name__ attribute raises
    ValueError.
    """

    # Create a callable object without __name__ attribute
    class CallableWithoutName:
        def __call__(self):
            pass  # pragma: no cover

    callable_obj = CallableWithoutName()

    with pytest.raises(ValueError) as exc_info:
        registry.register(callable_obj)

    assert "Object has no __name__ attribute and no name was provided" in str(
        exc_info.value
    )


def test_is_registered_with_function():
    """
    Test is_registered with a function that has __is_expressions_type__
    attribute.
    """

    def test_func():
        pass  # pragma: no cover

    setattr(test_func, "__is_expressions_type__", True)

    assert registry.is_registered(test_func)


def test_get_all_with_modules():
    """Test that get_all properly includes registered modules."""
    register_module("math", alias="math_alias")
    register_module("random")

    all_objects = registry.get_all()
    assert "math_alias" in all_objects
    assert "random" in all_objects


def test_register_with_invalid_name():
    """Test that registering with an invalid name type raises ValueError."""

    class TestObject:
        pass  # pragma: no cover

    with pytest.raises(ValueError) as exc_info:
        registry.register()(TestObject())  # Pass an object without __name__

    assert "No name provided and object has no __name__ attribute" in str(
        exc_info.value
    )


def test_register_in_namespace_type_error():
    """
    Test that registering in namespace with a non-string name raises TypeError.
    """
    namespace = {}
    with pytest.raises(TypeError) as exc_info:
        # Pass a non-string name
        registry._register_in_namespace(123, "value", namespace)  # type: ignore

    assert "Name must be a string" in str(exc_info.value)


def test_register_in_namespace_empty_name():
    """
    Test that registering in namespace with an empty name raises ValueError.
    """
    namespace = {}
    with pytest.raises(ValueError) as exc_info:
        registry._register_in_namespace(
            "", "value", namespace
        )  # Pass an empty string

    assert "Name cannot be empty" in str(exc_info.value)


def test_is_registered_with_non_function():
    """Test is_registered with a non-function object."""

    class TestClass:
        pass

    obj = TestClass()
    assert registry.is_registered(
        obj
    )  # Should return True for non-function objects


def test_get_all_with_mixed_content():
    """
    Test that get_all properly handles mixed content
    (modules, objects, namespaces).
    """
    # Register a module
    register_module("math")

    # Register a function in a namespace
    @register("utils.helper")
    def helper():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    # Register a direct object
    @register
    def direct_func():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    all_objects = registry.get_all()
    assert "math" in all_objects
    assert isinstance(all_objects["utils"], NamespaceProxy)
    assert callable(all_objects["utils"].helper)
    assert callable(all_objects["direct_func"])


def test_is_registered_with_string_name():
    """Test is_registered with a string name."""

    @register("test.func")
    def test_func():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    assert registry.is_registered("test.func")
    assert not registry.is_registered("non.existent.func")


def test_get_all_with_non_dict_values():
    """Test that get_all properly handles non-dict values."""

    @register("test.value")
    def test_func():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    @register("direct_value")
    def direct_func():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    register_module("math")

    all_objects = registry.get_all()
    assert callable(all_objects["test"].value)  # Nested function
    assert callable(all_objects["direct_value"])  # Direct function
    assert hasattr(all_objects["math"], "sqrt")  # Module


def test_register_in_namespace_with_empty_parts():
    """Test registering in namespace with empty parts in the name."""
    namespace = {}
    with pytest.raises(ValueError) as exc_info:
        registry._register_in_namespace("test..func", "value", namespace)

    assert "Name cannot contain empty parts" in str(exc_info.value)


def test_register_in_nested_namespace_with_empty_parts():
    """Test registering in nested namespace with empty parts in the name."""
    namespace = {"test": {"nested": {}}}
    with pytest.raises(ValueError) as exc_info:
        registry._register_in_namespace("test.nested..func", "value", namespace)

    assert "Name cannot contain empty parts" in str(exc_info.value)
