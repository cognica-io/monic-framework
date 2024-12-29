#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

# pylint: disable=unused-variable,protected-access

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
    UnsupportedUnpackingError,
)
from monic.expressions.registry import registry, NamespaceProxy


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    registry.reset()
    yield


def test_namespace_proxy():
    """Test NamespaceProxy functionality."""
    # Test NamespaceProxy
    namespace = {"a": 1, "nested": {"b": 2}}
    proxy = NamespaceProxy(namespace)

    # Test __getattr__ success and nested access
    assert proxy.a == 1
    assert proxy.nested.b == 2

    # Test __getattr__ failure
    with pytest.raises(AttributeError):
        _ = proxy.non_existent


def test_registry_basic_operations():
    """Test basic registry operations."""

    # Test register with no name
    class NoName:
        pass  # pragma: no cover

    with pytest.raises(
        ValueError,
        match="No name provided and object has no __name__ attribute",
    ):
        registry.register()(NoName())

    # Test register with existing name
    @registry.register("test.func")
    def test_func():
        pass  # pragma: no cover

    with pytest.raises(ValueError, match="is already registered in namespace"):

        @registry.register("test.func")
        def another_func():
            pass  # pragma: no cover

    # Test register_module with import error
    with pytest.raises(ImportError):
        registry.register_module("non_existent_module")

    # Test register_module with existing module
    registry.register_module("os", "sys_os")
    with pytest.raises(ValueError):
        registry.register_module("os", "sys_os")


def test_registry_get_operations():
    """Test registry get operations."""
    # Test get with non-existent name
    with pytest.raises(KeyError):
        registry.get("non_existent")

    # Test get with nested non-existent name
    with pytest.raises(KeyError):
        registry.get("nested.non_existent")

    # Test is_registered with various inputs
    @registry.register("test.func")
    def test_func():
        pass  # pragma: no cover

    assert registry.is_registered("test.func") is True
    assert registry.is_registered("non_existent") is False
    assert registry.is_registered(test_func) is True


def test_registry_nested_namespace():
    """Test registry nested namespace operations."""
    # Test register with name conflict in nested namespace
    registry._objects["conflict"] = "value"
    with pytest.raises(ValueError):
        registry._register_in_namespace(
            "conflict.nested", "new_value", registry._objects
        )

    # Test register with nested namespace creation
    @registry.register("deep.nested.func")
    def nested_func():
        pass  # pragma: no cover

    assert registry.is_registered("deep.nested.func") is True

    # Test register with nested namespace conflict
    registry._objects["ns"] = {"sub": 42}
    with pytest.raises(ValueError):
        registry._register_in_namespace(
            "ns.sub.deeper", "value", registry._objects
        )


def test_registry_callable_objects():
    """Test registry operations with callable objects."""

    # Test register with callable object without __name__
    class CallableWithoutName:
        def __call__(self):
            pass  # pragma: no cover

    with pytest.raises(ValueError):
        registry.register(CallableWithoutName())

    # Test register with direct function call (no decorator)
    def direct_func():
        pass  # pragma: no cover

    registry.register("direct")(direct_func)
    assert registry.is_registered("direct") is True

    # Test register with object that has no __name__ attribute
    class CustomCallable:
        def __call__(self, *args, **kwargs):
            pass  # pragma: no cover

    obj = CustomCallable()
    with pytest.raises(ValueError):
        registry.register(obj)


def test_registry_module_operations():
    """Test registry module operations."""
    # Test register_module with nested alias
    math_module = registry.register_module("math", "math.core")
    assert registry.get("math.core") == math_module

    # Test register_module with default alias
    json_module = registry.register_module("json")
    assert json_module == registry._modules["json"]

    # Test register_module with nested name conflict
    registry._objects["math"] = {"nested": 42}
    with pytest.raises(ValueError):
        registry.register_module("os", "math.nested")


def test_registry_get_all():
    """Test registry get_all functionality."""

    @registry.register("deep.nested.func")
    def nested_func():
        pass  # pragma: no cover

    registry.register_module("os", "sys_os")

    all_objects = registry.get_all()
    assert isinstance(all_objects["deep"], NamespaceProxy)
    assert all_objects["sys_os"]


def test_registry_non_function_objects():
    """Test registry operations with non-function objects."""

    # Test register with non-function object
    class TestClass:
        pass  # pragma: no cover

    obj = TestClass()
    registry.register("test.obj")(obj)
    assert registry.is_registered(obj) is True

    # Test is_registered with non-function object
    class NonCallable:
        pass  # pragma: no cover

    obj = NonCallable()
    registry.register("non_callable")(obj)
    assert registry.is_registered(obj) is True

    # Test is_registered with non-function object in _objects
    registry._objects["non_func"] = 42
    assert registry.is_registered("non_func") is True


def test_registry_nested_name_conflicts():
    """Test registry nested name conflict handling."""
    # Test register with nested name that exists as a non-dict
    registry._objects["parent"] = 42
    with pytest.raises(ValueError):
        registry._register_object("parent.child", lambda: None)

    # Test register with name conflict in nested namespace
    registry._objects["ns2"] = {"sub": {}}
    registry._objects["ns2"]["sub"]["func"] = lambda: None
    with pytest.raises(ValueError):
        registry._register_object("ns2.sub.func", lambda: None)

    # Test register with name conflict in intermediate namespace
    registry._objects["conflict2"] = {"nested": 42}
    with pytest.raises(ValueError):
        registry._register_object("conflict2.nested.func", lambda: None)


def test_registry_nested_name_access():
    """Test registry nested name access."""

    # Test is_registered with nested name in _objects
    def inner_func():
        pass  # pragma: no cover

    setattr(inner_func, "__is_expressions_type__", True)
    registry._objects["nested_obj"] = {"func": inner_func}
    assert registry.is_registered("nested_obj.func") is True

    # Test is_registered with non-function object in nested namespace
    registry._objects["nested_obj2"] = {"value": 42}
    assert registry.is_registered("nested_obj2.value") is True

    # Test is_registered with function in nested namespace
    def inner_func2():
        pass  # pragma: no cover

    registry._objects["nested_obj3"] = {"func": inner_func2}
    assert registry.is_registered("nested_obj3.func") is False


def test_registry_non_existent_names():
    """Test registry operations with non-existent names."""
    # Test is_registered with non-existent nested name
    assert registry.is_registered("non_existent.func") is False

    # Test is_registered with non-existent intermediate namespace
    assert registry.is_registered("non_existent.nested.func") is False

    # Test is_registered with non-existent part in nested namespace
    registry._objects["partial"] = {"nested": {}}
    assert registry.is_registered("partial.nested.non_existent") is False


def test_registry_type_errors():
    """Test registry type error handling."""
    # Test register with non-dict intermediate namespace
    registry._objects["invalid"] = 42
    with pytest.raises(ValueError):
        registry._register_object("invalid.nested.func", lambda: None)

    # Test is_registered with non-dict intermediate namespace
    with pytest.raises(TypeError):
        registry.get("invalid.nested.func")


def test_unpacking_basic():
    """Test basic unpacking operations."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test basic unpacking with starred expressions
    code = """
data = [1, 2, 3, 4, 5]
first, *middle, last = data
result = [first, middle, last]
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [1, [2, 3, 4], 5]


def test_unpacking_errors():
    """Test unpacking error cases."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test too many values
    code = """
try:
    a, b = [1, 2, 3]
    result = "should not reach here"
except ValueError as e:
    if str(e) == "Too many values to unpack":
        result = "too many values"
    else:
        result = str(e)
result
"""
    tree = parser.parse(code)
    with pytest.raises(UnsupportedUnpackingError) as exc_info:
        interpreter.execute(tree)
    assert str(exc_info.value) == "Too many values to unpack"

    # Test not enough values
    code = """
try:
    a, b, c = [1, 2]
    result = "should not reach here"
except ValueError as e:
    if str(e) == "Not enough values to unpack":
        result = "not enough values"
    else:
        result = str(e)
result
"""
    tree = parser.parse(code)
    with pytest.raises(UnsupportedUnpackingError) as exc_info:
        interpreter.execute(tree)
    assert str(exc_info.value) == "Not enough values to unpack"


def test_match_basic_sequence():
    """Test basic match statement sequence patterns."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Add required functions to the interpreter's environment
    interpreter.global_env.update(
        {
            "len": len,
            "str": str,
        }
    )

    code = """
def analyze_sequence(seq):
    match seq:
        case [first, *middle, last] if len(middle) > 1:
            return f"Long sequence: {first}...{last}"
        case [x, y]:
            return f"Pair: {x}, {y}"
        case [single]:
            return f"Single item: {single}"
        case []:
            return "Empty"
        case _:
            return "Not a sequence"

results = []
results.append(analyze_sequence([1, 2, 3, 4, 5]))
results.append(analyze_sequence([1, 2]))
results.append(analyze_sequence([42]))
results.append(analyze_sequence([]))
results.append(analyze_sequence("not a list"))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "Long sequence: 1...5",
        "Pair: 1, 2",
        "Single item: 42",
        "Empty",
        "Not a sequence",
    ]


def test_match_nested_sequence():
    """Test nested match statement sequence patterns."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Add required functions to the interpreter's environment
    interpreter.global_env.update(
        {
            "len": len,
            "str": str,
        }
    )

    code = """
def analyze_matrix(matrix):
    match matrix:
        case [[x, y], [a, b]]:
            return f"2x2 matrix: {x},{y} | {a},{b}"
        case [[]]:
            return "Empty matrix"
        case _:
            return "Invalid matrix"

results = []
results.append(analyze_matrix([[1, 2], [3, 4]]))
results.append(analyze_matrix([[1, 2], [3]]))  # Uneven rows
results.append(analyze_matrix([[]]))
results.append(analyze_matrix([1, 2, 3]))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "2x2 matrix: 1,2 | 3,4",
        "Invalid matrix",
        "Empty matrix",
        "Invalid matrix",
    ]
