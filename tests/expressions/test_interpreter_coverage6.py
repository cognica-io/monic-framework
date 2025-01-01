#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

# pylint: disable=protected-access

import pytest

from monic.expressions.registry import Registry, registry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    registry.reset()
    yield


def test_bind_default_without_name():
    """Test binding an object without explicit name to default registry."""
    reg = Registry()

    @reg.bind_default
    def test_func():
        return 42  # pragma: no cover

    assert callable(reg._default_objects["test_func"])


def test_bind_default_with_name():
    """Test binding an object with explicit name to default registry."""
    reg = Registry()

    @reg.bind_default("answer")
    def get_answer():
        return 42  # pragma: no cover

    assert callable(reg._default_objects["answer"])


def test_bind_default_with_nested_name():
    """Test binding objects with nested names to default registry."""
    reg = Registry()

    @reg.bind_default("math.functions.add")
    def add(x, y):
        return x + y  # pragma: no cover

    assert callable(reg._default_objects["math"]["functions"]["add"])


def test_bind_default_module():
    """Test binding a module to default registry."""
    reg = Registry()
    reg.bind_default_module("json", "json_util")
    assert "json_util" in reg._default_modules


def test_bind_default_module_with_nested_name():
    """Test binding a module with nested name to default registry."""
    reg = Registry()
    reg.bind_default_module("json", "util.json")
    assert "json" in reg._default_objects["util"]


def test_bind_default_module_duplicate():
    """Test binding the same module twice to default registry."""
    reg = Registry()
    reg.bind_default_module("json", "json_util")

    with pytest.raises(ValueError) as exc_info:
        reg.bind_default_module("json", "json_util")
    assert "Module 'json_util' is already bound" in str(exc_info.value)


def test_bind_default_module_nested_name_conflict():
    """
    Test binding a module with conflicting nested name to default registry.
    """
    reg = Registry()
    reg._default_objects["util"] = 42

    with pytest.raises(ValueError) as exc_info:
        reg.bind_default_module("json", "util.json")
    assert "is already bound as a non-namespace" in str(exc_info.value)


def test_bind_default_module_import_error():
    """Test binding a non-existent module to default registry."""
    reg = Registry()

    with pytest.raises(ImportError) as exc_info:
        reg.bind_default_module("non_existent_module")
    assert "Failed to import module 'non_existent_module'" in str(
        exc_info.value
    )


def test_bind_default_object_without_name():
    """Test binding an object without name to default registry."""
    reg = Registry()

    with pytest.raises(ValueError) as exc_info:
        reg.bind_default()(object())
    assert "No name provided and object has no __name__ attribute" in str(
        exc_info.value
    )


def test_bind_default_object_with_name_conflict():
    """Test binding an object with conflicting name to default registry."""
    reg = Registry()
    reg._bind_default_object("test", 42)

    with pytest.raises(ValueError) as exc_info:
        reg._bind_default_object("test", 43)
    assert "'test' is already bound in namespace" in str(exc_info.value)


def test_bind_default_object_with_nested_name_conflict():
    """
    Test binding an object with conflicting nested name to default registry.
    """
    reg = Registry()
    reg._default_objects["math"] = 42

    with pytest.raises(ValueError) as exc_info:
        reg._bind_default_object("math.add", lambda x, y: x + y)
    assert "'math' is already bound as a non-namespace" in str(exc_info.value)


def test_get_from_namespace_error():
    """Test getting a non-existent name from namespace."""
    reg = Registry()

    with pytest.raises(KeyError) as exc_info:
        reg._get_from_namespace("non_existent", reg._default_objects)
    assert "Name 'non_existent' is not defined" in str(exc_info.value)


def test_get_from_namespace_nested_error():
    """Test getting a non-existent nested name from namespace."""
    reg = Registry()
    reg._default_objects["math"] = {"functions": {}}

    with pytest.raises(KeyError) as exc_info:
        reg._get_from_namespace(
            "math.functions.non_existent", reg._default_objects
        )
    assert "Name 'math.functions.non_existent' is not defined" in str(
        exc_info.value
    )


def test_bind_default_callable_without_name():
    """
    Test binding a callable without __name__ attribute to default registry.
    """
    reg = Registry()

    class CallableWithoutName:
        def __call__(self):
            return 42  # pragma: no cover

    with pytest.raises(ValueError) as exc_info:
        reg.bind_default(CallableWithoutName())
    assert "Object has no __name__ attribute and no name was provided" in str(
        exc_info.value
    )


def test_bind_default_module_with_empty_name():
    """Test binding a module with empty name to default registry."""
    reg = Registry()

    with pytest.raises(ValueError) as exc_info:
        reg._bind_in_namespace("", None, reg._default_objects)
    assert "Name cannot be empty" in str(exc_info.value)


def test_bind_default_module_with_empty_parts():
    """Test binding a module with empty parts in name to default registry."""
    reg = Registry()

    with pytest.raises(ValueError) as exc_info:
        reg._bind_in_namespace("util..json", None, reg._default_objects)
    assert "Name cannot contain empty parts" in str(exc_info.value)


def test_bind_default_module_with_non_string_name():
    """Test binding a module with non-string name to default registry."""
    reg = Registry()

    with pytest.raises(TypeError) as exc_info:
        reg._bind_in_namespace(123, None, reg._default_objects)  # type: ignore
    assert "Name must be a string" in str(exc_info.value)


def test_bind_callable_without_name():
    """Test binding a callable without __name__ attribute."""
    reg = Registry()

    class CallableWithoutName:
        def __call__(self):
            return 42  # pragma: no cover

    with pytest.raises(ValueError) as exc_info:
        reg.bind(CallableWithoutName())
    assert "Object has no __name__ attribute and no name was provided" in str(
        exc_info.value
    )


def test_bind_with_name_conflict():
    """Test binding with name conflict."""
    reg = Registry()

    @reg.bind("test")
    def func1():
        return 42  # pragma: no cover

    with pytest.raises(ValueError) as exc_info:

        @reg.bind("test")
        def func2():
            return 43  # pragma: no cover

    assert "'test' is already bound in namespace" in str(exc_info.value)


def test_bind_module_with_name_conflict():
    """Test binding module with name conflict."""
    reg = Registry()
    reg.bind_module("json", "test")

    with pytest.raises(ValueError) as exc_info:
        reg.bind_module("time", "test")
    assert "Module 'test' is already bound" in str(exc_info.value)


def test_bind_module_with_nested_name_conflict():
    """Test binding module with nested name conflict."""
    reg = Registry()
    reg._objects["util"] = 42

    with pytest.raises(ValueError) as exc_info:
        reg.bind_module("json", "util.json")
    assert "'util' is already bound as a non-namespace" in str(exc_info.value)


def test_bind_default_with_name_conflict():
    """Test binding default with name conflict."""
    reg = Registry()

    @reg.bind_default("test")
    def func1():
        return 42  # pragma: no cover

    with pytest.raises(ValueError) as exc_info:

        @reg.bind_default("test")
        def func2():
            return 43  # pragma: no cover

    assert "'test' is already bound in namespace" in str(exc_info.value)


def test_bind_with_no_name():
    """Test binding with no name."""
    reg = Registry()

    class CallableWithoutName:
        def __call__(self):
            return 42  # pragma: no cover

    with pytest.raises(ValueError) as exc_info:
        reg.bind()(CallableWithoutName())
    assert "No name provided and object has no __name__ attribute" in str(
        exc_info.value
    )


def test_bind_with_nested_name_conflict():
    """Test binding with nested name conflict."""
    reg = Registry()
    reg._objects["math"] = 42

    with pytest.raises(ValueError) as exc_info:

        @reg.bind("math.add")
        def add(x, y):
            return x + y  # pragma: no cover

    assert "'math' is already bound as a non-namespace" in str(exc_info.value)


def test_bind_module_with_import_error():
    """Test binding module with import error."""
    reg = Registry()

    with pytest.raises(ImportError) as exc_info:
        reg.bind_module("non_existent_module")
    assert "Failed to import module 'non_existent_module'" in str(
        exc_info.value
    )


def test_bind_default_with_no_name():
    """Test binding default with no name."""
    reg = Registry()

    class CallableWithoutName:
        def __call__(self):
            return 42  # pragma: no cover

    with pytest.raises(ValueError) as exc_info:
        reg.bind_default()(CallableWithoutName())
    assert "No name provided and object has no __name__ attribute" in str(
        exc_info.value
    )


def test_bind_with_name_and_no_name_attr():
    """Test binding with name but no __name__ attribute."""
    reg = Registry()

    class CallableWithoutName:
        def __call__(self):
            return 42  # pragma: no cover

    _ = reg.bind("test")(CallableWithoutName())
    assert callable(reg._objects["test"])


def test_bind_with_nested_name():
    """Test binding with nested name."""
    reg = Registry()

    @reg.bind("math.functions.add")
    def add(x, y):
        return x + y  # pragma: no cover

    assert callable(reg._objects["math"]["functions"]["add"])


def test_bind_module_with_nested_name():
    """Test binding module with nested name."""
    reg = Registry()
    reg.bind_module("json", "util.json")
    assert "json" in reg._objects["util"]


def test_bind_default_with_name_and_no_name_attr():
    """Test binding default with name but no __name__ attribute."""
    reg = Registry()

    class CallableWithoutName:
        def __call__(self):
            return 42  # pragma: no cover

    _ = reg.bind_default("test")(CallableWithoutName())
    assert callable(reg._default_objects["test"])


def test_bind_with_name_and_function():
    """Test binding with name and function."""
    reg = Registry()

    def add(x, y):
        return x + y  # pragma: no cover

    obj = reg.bind("math.add")(add)
    assert callable(reg._objects["math"]["add"])
    assert hasattr(obj, "__is_expressions_type__")


def test_bind_default_with_name_and_function():
    """Test binding default with name and function."""
    reg = Registry()

    def add(x, y):
        return x + y  # pragma: no cover

    obj = reg.bind_default("math.add")(add)
    assert callable(reg._default_objects["math"]["add"])
    assert hasattr(obj, "__is_expressions_type__")


def test_bind_module_with_no_alias():
    """Test binding module with no alias."""
    reg = Registry()
    module = reg.bind_module("json")
    assert module == reg._modules["json"]


def test_bind_default_module_with_no_alias():
    """Test binding default module with no alias."""
    reg = Registry()
    module = reg.bind_default_module("json")
    assert module == reg._default_modules["json"]


def test_bind_with_function_no_name_attr():
    """Test binding with function that has no __name__ attribute."""
    reg = Registry()

    # Create a callable object without __name__ attribute
    def create_callable():
        class CallableWithoutName:
            def __call__(self):
                return 42  # pragma: no cover

        return CallableWithoutName()

    obj = create_callable()

    with pytest.raises(ValueError) as exc_info:
        reg.bind(obj)
    assert "Object has no __name__ attribute and no name was provided" in str(
        exc_info.value
    )


def test_bind_default_with_function_no_name_attr():
    """Test binding default with function that has no __name__ attribute."""
    reg = Registry()

    # Create a callable object without __name__ attribute
    def create_callable():
        class CallableWithoutName:
            def __call__(self):
                return 42  # pragma: no cover

        return CallableWithoutName()

    obj = create_callable()

    with pytest.raises(ValueError) as exc_info:
        reg.bind_default(obj)
    assert "Object has no __name__ attribute and no name was provided" in str(
        exc_info.value
    )


def test_get_from_namespace_nested():
    """Test getting from nested namespace."""
    reg = Registry()
    reg._objects["math"] = {"functions": {"add": lambda x, y: x + y}}
    result = reg._get_from_namespace("math.functions.add", reg._objects)
    assert callable(result)


def test_get_from_namespace_not_found():
    """Test getting from namespace when name not found."""
    reg = Registry()
    reg._objects["math"] = {"functions": {}}
    with pytest.raises(KeyError) as exc_info:
        reg._get_from_namespace("math.functions.add", reg._objects)
    assert "Name 'math.functions.add' is not defined" in str(exc_info.value)


def test_get_all_with_modules():
    """Test getting all objects including modules."""
    reg = Registry()

    # Add a module to _modules
    module = reg.bind_module("json")

    # Add a module to _default_modules
    default_module = reg.bind_default_module("time")

    # Add objects to _objects and _default_objects
    reg._objects["test"] = 42
    reg._default_objects["default_test"] = 43

    # Add nested objects
    reg._objects["math"] = {"add": lambda x, y: x + y}
    reg._default_objects["utils"] = {"helper": lambda x: x * 2}

    # Add a module with nested name to _objects
    os_path_module = reg.bind_module("os.path", "os.path")

    # Add a module directly to _modules
    sys_module = reg.bind_module("sys")

    # Add another module directly to _modules to ensure update is called
    platform_module = reg.bind_module("platform")

    # Add a module that will be overridden
    reg._modules["override"] = module
    reg._objects["override"] = 42

    # Get all objects
    result = reg.get_all()

    # Check that everything is included
    assert result["json"] == module
    assert result["time"] == default_module
    assert result["test"] == 42
    assert result["default_test"] == 43
    assert hasattr(result["math"], "_namespace")  # Check if it's a proxy
    assert hasattr(result["utils"], "_namespace")  # Check if it's a proxy
    assert callable(result["math"].add)
    assert callable(result["utils"].helper)
    assert hasattr(result["os"], "_namespace")  # Check if it's a proxy
    assert result["os"].path == os_path_module  # Check the actual module
    assert result["sys"] == sys_module  # Check module in _modules
    assert (
        result["platform"] == platform_module
    )  # Check another module in _modules
    assert result["override"] == module  # Check that _modules takes precedence

    # Check that _modules is actually used in the result
    assert len(reg._modules) > 0
    assert all(name in result for name in reg._modules)

    # Check that the result contains all modules
    for name, value in reg._modules.items():
        assert result[name] == value

    # Check that _modules is updated after _objects
    assert len(result) >= len(reg._objects) + len(reg._modules)

    # Check that _modules is updated correctly
    modules_dict = dict(reg._modules)
    for name, value in modules_dict.items():
        assert result[name] == value
