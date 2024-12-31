#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

# pylint: disable=protected-access

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
    monic_bind,
    monic_bind_module,
)
from monic.expressions.registry import registry, NamespaceProxy


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    registry.reset()
    yield


def test_bound_function():
    """Test using bound functions."""

    @monic_bind
    def custom_add(x, y):  # pylint: disable=unused-variable
        """Custom addition function."""
        return x + y

    @monic_bind("multiply")
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


def test_bound_class():
    """Test using bound class."""

    @monic_bind
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


def test_bind_without_name():
    """Test binding an object without explicit name."""

    @monic_bind
    def test_func():  # pylint: disable=unused-variable
        return 42

    code = "result = test_func()"
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result") == 42


def test_bind_with_name():
    """Test binding an object with explicit name."""

    @monic_bind("answer")
    def get_answer():  # pylint: disable=unused-variable
        return 42

    code = "result = answer()"
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result") == 42


def test_bind_with_nested_name():
    """Test binding objects with nested names."""

    @monic_bind("math.functions.add")
    def add(x, y):  # pylint: disable=unused-variable
        return x + y

    @monic_bind("math.functions.multiply")
    def multiply(x, y):  # pylint: disable=unused-variable
        return x * y

    @monic_bind("math.constants")
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


def test_bind_nested_name_conflict():
    """Test that binding conflicting nested names raises an error."""

    @monic_bind("math.functions.add")
    def add1(x, y):  # pylint: disable=unused-variable
        return x + y  # pragma: no cover

    # Try to bind another function with the same nested name
    with pytest.raises(ValueError) as exc_info:

        @monic_bind("math.functions.add")
        def add2(x, y):  # pylint: disable=unused-variable
            return x + y  # pragma: no cover

    assert "is already bound in namespace" in str(exc_info.value)


def test_bind_nested_name_non_namespace_conflict():
    """Test conflict between nested name and non-namespace object."""

    @monic_bind("math")
    def math_func():  # pylint: disable=unused-variable
        return 42  # pragma: no cover

    # Try to bind another function with the same nested name
    with pytest.raises(ValueError) as exc_info:

        @monic_bind("math.functions.add")
        def add(x, y):  # pylint: disable=unused-variable
            return x + y  # pragma: no cover

    assert "is already bound as a non-namespace" in str(exc_info.value)


def test_bind_both_syntaxes():
    """Test both decorator syntaxes work correctly."""

    @monic_bind
    def func1():  # pylint: disable=unused-variable
        return 1

    @monic_bind()
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


def test_bind_class_both_syntaxes():
    """Test both decorator syntaxes work correctly with classes."""

    @monic_bind
    class Class1:  # pylint: disable=unused-variable
        value = 1

    @monic_bind()
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


def test_bind_module():
    """Test binding and using a module."""
    # Bind math module
    monic_bind_module("math")

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


def test_bind_module_with_alias():
    """Test binding a module with an alias."""
    # Bind random module with alias
    monic_bind_module("random", alias="rand")

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


def test_bind_module_submodule():
    """Test binding a module with submodules."""
    # Bind collections.abc module
    monic_bind_module("collections.abc", alias="collabc")

    code = """
    is_sequence = collabc.Sequence
    result = isinstance([1, 2, 3], is_sequence)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    assert result is True


def test_bind_module_with_nested_name():
    """Test binding modules with nested names."""
    # Bind urllib and its parse submodule
    monic_bind_module("urllib", alias="url")
    monic_bind_module("urllib.parse", alias="url.parse")

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


def test_bind_non_nested_name_conflict():
    """
    Test that binding a non-nested name that already exists raises an error.
    """
    namespace = {}
    registry._bind_in_namespace("test", 42, namespace)

    with pytest.raises(ValueError) as exc_info:
        registry._bind_in_namespace("test", 43, namespace)

    assert "Name 'test' is already bound" in str(exc_info.value)


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


def test_bind_object_without_name():
    """
    Test that binding an object without a name and no __name__ attribute
    raises ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        registry.bind()(object())

    assert "No name provided and object has no __name__ attribute" in str(
        exc_info.value
    )


def test_bind_module_import_error():
    """Test that binding a non-existent module raises ImportError."""
    with pytest.raises(ImportError) as exc_info:
        monic_bind_module("non_existent_module")

    assert "Failed to import module 'non_existent_module'" in str(
        exc_info.value
    )


def test_bind_module_duplicate():
    """Test that binding the same module twice raises ValueError."""
    monic_bind_module("math")

    with pytest.raises(ValueError) as exc_info:
        monic_bind_module("math")

    assert "Module 'math' is already bound" in str(exc_info.value)


def test_get_all_with_nested_namespaces():
    """Test that get_all properly handles nested namespaces."""

    @monic_bind("math.functions.add")
    def add(x, y):  # pylint: disable=unused-variable
        return x + y  # pragma: no cover

    @monic_bind("math.constants")
    class Constants:  # pylint: disable=unused-variable
        PI = 3.14159

    all_objects = registry.get_all()
    assert isinstance(all_objects["math"], NamespaceProxy)
    assert isinstance(all_objects["math"].functions, NamespaceProxy)
    assert callable(all_objects["math"].functions.add)
    assert all_objects["math"].constants.PI == 3.14159


def test_bind_callable_without_name():
    """
    Test that binding a callable without __name__ attribute raises
    ValueError.
    """

    # Create a callable object without __name__ attribute
    class CallableWithoutName:
        def __call__(self):
            pass  # pragma: no cover

    callable_obj = CallableWithoutName()

    with pytest.raises(ValueError) as exc_info:
        registry.bind(callable_obj)

    assert "Object has no __name__ attribute and no name was provided" in str(
        exc_info.value
    )


def test_is_bound_with_function():
    """
    Test is_bound with a function that has __is_expressions_type__
    attribute.
    """

    def test_func():
        pass  # pragma: no cover

    setattr(test_func, "__is_expressions_type__", True)

    assert registry.is_bound(test_func)


def test_get_all_with_modules():
    """Test that get_all properly includes bound modules."""
    monic_bind_module("math", alias="math_alias")
    monic_bind_module("random")

    all_objects = registry.get_all()
    assert "math_alias" in all_objects
    assert "random" in all_objects


def test_bind_with_invalid_name():
    """Test that binding with an invalid name type raises ValueError."""

    class TestObject:
        pass  # pragma: no cover

    with pytest.raises(ValueError) as exc_info:
        registry.bind()(TestObject())  # Pass an object without __name__

    assert "No name provided and object has no __name__ attribute" in str(
        exc_info.value
    )


def test_bind_in_namespace_type_error():
    """
    Test that binding in namespace with a non-string name raises TypeError.
    """
    namespace = {}
    with pytest.raises(TypeError) as exc_info:
        # Pass a non-string name
        registry._bind_in_namespace(123, "value", namespace)  # type: ignore

    assert "Name must be a string" in str(exc_info.value)


def test_bind_in_namespace_empty_name():
    """
    Test that binding in namespace with an empty name raises ValueError.
    """
    namespace = {}
    with pytest.raises(ValueError) as exc_info:
        registry._bind_in_namespace(
            "", "value", namespace  # Pass an empty string
        )

    assert "Name cannot be empty" in str(exc_info.value)


def test_is_bound_with_non_function():
    """Test is_bound with a non-function object."""

    class TestClass:
        pass

    obj = TestClass()
    assert registry.is_bound(obj)  # Should return True for non-function objects


def test_get_all_with_mixed_content():
    """
    Test that get_all properly handles mixed content
    (modules, objects, namespaces).
    """
    # Bind a module
    monic_bind_module("math")

    # Bind a function in a namespace
    @monic_bind("utils.helper")
    def helper():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    # Bind a direct object
    @monic_bind
    def direct_func():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    all_objects = registry.get_all()
    assert "math" in all_objects
    assert isinstance(all_objects["utils"], NamespaceProxy)
    assert callable(all_objects["utils"].helper)
    assert callable(all_objects["direct_func"])


def test_is_bound_with_string_name():
    """Test is_bound with a string name."""

    @monic_bind("test.func")
    def test_func():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    assert registry.is_bound("test.func")
    assert not registry.is_bound("non.existent.func")


def test_get_all_with_non_dict_values():
    """Test that get_all properly handles non-dict values."""

    @monic_bind("test.value")
    def test_func():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    @monic_bind("direct_value")
    def direct_func():  # pylint: disable=unused-variable
        pass  # pragma: no cover

    monic_bind_module("math")

    all_objects = registry.get_all()
    assert callable(all_objects["test"].value)  # Nested function
    assert callable(all_objects["direct_value"])  # Direct function
    assert hasattr(all_objects["math"], "sqrt")  # Module


def test_bind_in_namespace_with_empty_parts():
    """Test binding in namespace with empty parts in the name."""
    namespace = {}
    with pytest.raises(ValueError) as exc_info:
        registry._bind_in_namespace("test..func", "value", namespace)

    assert "Name cannot contain empty parts" in str(exc_info.value)


def test_bind_in_nested_namespace_with_empty_parts():
    """Test binding in nested namespace with empty parts in the name."""
    namespace = {"test": {"nested": {}}}
    with pytest.raises(ValueError) as exc_info:
        registry._bind_in_namespace("test.nested..func", "value", namespace)

    assert "Name cannot contain empty parts" in str(exc_info.value)
