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


def test_unpacking_with_starred_expression():
    """Test unpacking with starred expressions."""
    code = """
    values = [1, 2, 3, 4, 5]
    first, *middle, last = values
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("first") == 1
    assert interpreter.get_name_value("middle") == [2, 3, 4]
    assert interpreter.get_name_value("last") == 5


def test_unpacking_with_multiple_starred_expressions():
    """Test that unpacking with multiple starred expressions raises an error."""
    code = """
    values = [1, 2, 3, 4, 5]
    first, *middle, *end, last = values
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(UnsupportedUnpackingError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "Cannot use multiple starred expressions in assignment" in str(
        exc_info.value
    )


def test_unpacking_with_insufficient_values():
    """Test unpacking with insufficient values."""
    code = """
    values = [1, 2]
    a, b, c = values
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(UnsupportedUnpackingError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "Not enough values to unpack" in str(exc_info.value)


def test_unpacking_with_too_many_values():
    """Test unpacking with too many values."""
    code = """
    values = [1, 2, 3]
    a, b = values
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(UnsupportedUnpackingError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "Too many values to unpack" in str(exc_info.value)


def test_unpacking_non_iterable():
    """Test unpacking a non-iterable value."""
    code = """
    value = 42
    a, b = value
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(UnsupportedUnpackingError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "Cannot unpack non-iterable value" in str(exc_info.value)


def test_unpacking_with_attribute_target():
    """Test unpacking with attribute target."""
    code = """
    class Point:
        def __init__(self):
            self.x = 0
            self.y = 0

    p = Point()
    p.x, p.y = (1, 2)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    p = interpreter.get_name_value("p")
    assert p.x == 1
    assert p.y == 2


def test_unpacking_with_subscript_target():
    """Test unpacking with subscript target."""
    code = """
    lst = [0, 0]
    temp = (1, 2)
    lst[0] = temp[0]
    lst[1] = temp[1]
    result = lst
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    assert result == [1, 2]


def test_list_comprehension_with_multiple_if():
    """Test list comprehension with multiple if conditions."""
    code = """
    numbers = range(10)
    evens_less_than_6 = [x for x in numbers if x % 2 == 0 if x < 6]
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("evens_less_than_6")
    assert result == [0, 2, 4]


def test_list_comprehension_with_multiple_for():
    """Test list comprehension with multiple for loops."""
    code = """
    matrix = [[1, 2], [3, 4]]
    flattened = [x for row in matrix for x in row]
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("flattened")
    assert result == [1, 2, 3, 4]


def test_set_comprehension_with_condition():
    """Test set comprehension with condition."""
    code = """
    numbers = range(5)
    squares = {x * x for x in numbers if x % 2 == 0}
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("squares")
    assert result == {0, 4, 16}


def test_dict_comprehension_with_condition():
    """Test dictionary comprehension with condition."""
    code = """
    numbers = range(5)
    squares = {x: x * x for x in numbers if x % 2 == 0}
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("squares")
    assert result == {0: 0, 2: 4, 4: 16}


def test_nested_comprehension():
    """Test nested comprehension."""
    code = """
    matrix = [[1, 2], [3, 4]]
    transposed = [[row[i] for row in matrix] for i in range(2)]
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("transposed")
    assert result == [[1, 3], [2, 4]]


def test_comprehension_with_unpacking():
    """Test comprehension with unpacking."""
    code = """
    pairs = [(1, 2), (3, 4)]
    sums = [x + y for x, y in pairs]
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("sums")
    assert result == [3, 7]


def test_function_call_with_kwargs_unpacking():
    """Test function call with kwargs unpacking."""
    code = """
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    kwargs = {"name": "World", "greeting": "Hi"}
    result = greet(**kwargs)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    assert result == "Hi, World!"


def test_function_call_with_invalid_kwargs():
    """Test function call with invalid kwargs."""
    code = """
    def greet(name):
        return f"Hello, {name}!"

    kwargs = {123: "World"}  # Invalid key type
    result = greet(**kwargs)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(TypeError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "Keywords must be strings" in str(exc_info.value)


def test_class_with_static_method():
    """Test class with static method."""
    code = """
    class MathUtils:
        @staticmethod
        def add(x, y):
            return x + y

    result = MathUtils.add(2, 3)
    instance = MathUtils()
    result2 = instance.add(4, 5)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result") == 5
    assert interpreter.get_name_value("result2") == 9


def test_class_with_class_method():
    """Test class with class method."""
    code = """
    class Counter:
        count = 0

        @classmethod
        def increment(cls):
            cls.count += 1
            return cls.count

    result1 = Counter.increment()
    result2 = Counter.increment()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result1") == 1
    assert interpreter.get_name_value("result2") == 2


def test_class_with_property():
    """Test class with property."""
    code = """
    class Circle:
        def __init__(self, radius):
            self._radius = radius

        def get_radius(self):
            return self._radius

        def set_radius(self, value):
            if value < 0:
                raise ValueError("Radius cannot be negative")
            self._radius = value

        radius = property(get_radius, set_radius)

    circle = Circle(5)
    radius1 = circle.radius
    circle.radius = 10
    radius2 = circle.radius
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("radius1") == 5
    assert interpreter.get_name_value("radius2") == 10


def test_class_with_super():
    """Test class with super() call."""
    code = """
    class Animal:
        def speak(self):
            return "Some sound"

    class Dog(Animal):
        def speak(self):
            return super().speak() + " Woof!"

    dog = Dog()
    result = dog.speak()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    assert result == "Some sound Woof!"


def test_registry_initialization():
    """Test registry initialization and attributes."""
    from monic.expressions.registry import Registry

    reg = Registry()
    reg.reset()


def test_registry_register_errors():
    """Test error handling in register method."""
    from monic.expressions.registry import Registry

    reg = Registry()
    error1 = ""
    error2 = ""

    try:
        reg.register()(None)  # Should raise ValueError
    except ValueError as e:
        error1 = str(e)

    class NoName:
        def __call__(self):
            return None

    obj = NoName()
    try:
        reg.register()(obj)  # Should raise ValueError
    except ValueError as e:
        error2 = str(e)

    assert "No name provided and object has no __name__ attribute" in error1
    assert "No name provided and object has no __name__ attribute" in error2


def test_registry_register_module_errors():
    """Test error handling in register_module method."""
    from monic.expressions.registry import Registry

    reg = Registry()
    error = ""

    try:
        reg.register_module("non_existent_module")  # Should raise ImportError
    except ImportError as e:
        error = str(e)

    assert "No module named" in error


def test_registry_get_all_errors():
    """Test error handling in get_all method."""
    from monic.expressions.registry import Registry

    reg = Registry()
    error = ""

    try:
        reg.get("non_existent_namespace")  # Should raise KeyError
    except KeyError as e:
        error = str(e)

    assert "non_existent_namespace" in str(error)


def test_nonlocal_declaration():
    """Test nonlocal declaration handling."""
    code = """
    def outer():
        x = 1
        def inner():
            nonlocal x
            x = 2
        inner()
        return x

    result = outer()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result") == 2


def test_nonlocal_declaration_errors():
    """Test nonlocal declaration error cases."""
    code = """
    def outer():
        def inner():
            nonlocal x  # Should raise SyntaxError
            x = 1
        inner()
    outer()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(SyntaxError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "No binding for nonlocal 'x' found in outer scopes" in str(
        exc_info.value
    )

    # Test nonlocal declaration at module level
    code = """
    def test():
        x = 1
    nonlocal x  # Should raise SyntaxError
    """
    with pytest.raises(SyntaxError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "nonlocal declaration not allowed at module level" in str(
        exc_info.value
    )


def test_name_deletion():
    """Test name deletion handling."""
    code = """
    x = 1
    del x
    try:
        y = x  # Should raise NameError
    except NameError as e:
        error = str(e)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    error = interpreter.get_name_value("error")
    assert "Name 'x' is not defined" in error


def test_name_deletion_errors():
    """Test name deletion error cases."""
    code = """
    try:
        del _  # Should raise SyntaxError
    except SyntaxError as e:
        error1 = str(e)

    try:
        del non_existent  # Should raise NameError
    except NameError as e:
        error2 = str(e)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    error1 = interpreter.get_name_value("error1")
    error2 = interpreter.get_name_value("error2")
    assert "Cannot delete special variable '_'" in error1
    assert "Name 'non_existent' is not defined" in error2


def test_name_scope_handling():
    """Test name scope handling."""
    code = """
    x = 1  # Global
    def outer():
        y = 3  # Local to outer
        def inner():
            nonlocal y
            y = 4  # Modifies outer's y
            z = 5  # Local to inner
        inner()
        return y  # Should be 4

    result1 = outer()
    result2 = x  # Should be 1 (unchanged)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    assert interpreter.get_name_value("result1") == 4
    assert interpreter.get_name_value("result2") == 1


def test_function_call_errors():
    """Test function call error cases."""
    code = """
    def func(a, b):
        return a + b

    try:
        result = func(1)  # Missing argument
    except TypeError as e:
        error1 = str(e)

    try:
        result = func(1, 2, 3)  # Too many arguments
    except TypeError as e:
        error2 = str(e)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    error1 = interpreter.get_name_value("error1")
    error2 = interpreter.get_name_value("error2")
    assert "missing required positional argument" in error1
    assert "takes 2 positional arguments but 3 were given" in error2


def test_function_definition_errors():
    """Test function definition error cases."""
    code = """
    def func(a, b=1, c):  # Non-default argument after default argument
        return a
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(SyntaxError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert (
        "parameter without a default follows parameter with a default"
        in str(exc_info.value)
    )

    # Test invalid parameter order
    code = """
    def func(a, **kwargs, b):  # Non-keyword argument after **kwargs
        return a
    """
    with pytest.raises(SyntaxError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "arguments cannot follow var-keyword argument" in str(exc_info.value)


def test_class_definition_errors():
    """Test class definition error cases."""
    code = """
    class Test:
        def __init__(self):
            pass

        def method(self):
            return self._private  # Access non-existent attribute

    t = Test()
    result = t.method()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(AttributeError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "'Test' object has no attribute '_private'" in str(exc_info.value)


def test_name_lookup_errors():
    """Test name lookup error cases."""
    code = """
    def outer():
        def inner():
            nonlocal x  # Should raise SyntaxError
            x = 1
        inner()
    outer()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(SyntaxError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "No binding for nonlocal 'x' found" in str(exc_info.value)


def test_name_deletion_complex():
    """Test complex name deletion scenarios."""
    code = """
    x = 1
    def outer():
        try:
            del x  # Should raise NameError (not in local scope)
        except NameError as e:
            return str(e)
    result = outer()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    assert "Name 'x' is not defined" in result


def test_function_definition_with_defaults():
    """Test function definition with default arguments."""
    code = """
    def func(a, b=1, *args, c=2, **kwargs):
        return (a, b, args, c, kwargs)

    result1 = func(0)
    result2 = func(0, 10, 20, 30, c=40, d=50)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result1 = interpreter.get_name_value("result1")
    result2 = interpreter.get_name_value("result2")
    assert result1 == (0, 1, (), 2, {})
    assert result2 == (0, 10, (20, 30), 40, {"d": 50})


def test_class_definition_with_metaclass():
    """Test class definition with metaclass."""
    code = """
    class Test:
        @classmethod
        def class_method(cls):
            return 'class method'

    result1 = Test.class_method()
    result2 = Test().class_method()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result1 = interpreter.get_name_value("result1")
    result2 = interpreter.get_name_value("result2")
    assert result1 == "class method"
    assert result2 == "class method"


def test_execution_error_handling():
    """Test execution error handling."""
    code = """
    def recurse():
        return recurse()
    result = recurse()  # Should raise RecursionError
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(RecursionError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "maximum recursion depth exceeded" in str(exc_info.value)


def test_attribute_access_errors():
    """Test attribute access error cases."""
    code = """
    class Test:
        pass

    t = Test()
    result = t.non_existent
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(AttributeError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "'Test' object has no attribute 'non_existent'" in str(
        exc_info.value
    )


def test_operator_errors():
    """Test operator error cases."""
    code = """
    class Test:
        pass

    t = Test()
    result = -t  # Unary minus on non-numeric type
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    with pytest.raises(TypeError) as exc_info:
        interpreter.execute(parser.parse(code))

    assert "bad operand type for unary -" in str(exc_info.value)


def test_class_definition_with_bases():
    """Test class definition with base classes."""
    code = """
    class Base1:
        def method1(self):
            return 1

    class Base2:
        def method2(self):
            return 2

    class Derived(Base1, Base2):
        def method3(self):
            return self.method1() + self.method2()

    d = Derived()
    result = d.method3()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    assert result == 3


def test_function_definition_with_annotations():
    """Test function definition with type annotations."""
    code = """
    def func(x: int, y: str = "default") -> str:
        return str(x) + y

    result = func(1)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    assert result == "1default"


def test_complex_name_lookup():
    """Test complex name lookup scenarios."""
    code = """
    x = 1
    def outer():
        x = 2
        def middle():
            nonlocal x  # Should refer to outer's x
            x = 3
            def inner():
                nonlocal x  # Should refer to middle's x
                x = 4
            inner()
            return x  # Should be 4
        return middle()
    result = outer()
    outer_x = x  # Should still be 1
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    outer_x = interpreter.get_name_value("outer_x")
    assert result == 4
    assert outer_x == 1


def test_complex_function_calls():
    """Test complex function call scenarios."""
    code = """
    def make_adder(x):
        def adder(y):
            return x + y
        return adder

    add5 = make_adder(5)
    result1 = add5(3)  # Should be 8

    def apply_twice(f, x):
        return f(f(x))

    result2 = apply_twice(lambda x: x * 2, 3)  # Should be 12
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result1 = interpreter.get_name_value("result1")
    result2 = interpreter.get_name_value("result2")
    assert result1 == 8
    assert result2 == 12


def test_complex_class_inheritance():
    """Test complex class inheritance scenarios."""
    code = """
    class A:
        def method(self):
            return 'A'

    class B(A):
        def method(self):
            return 'B' + super().method()

    class C(A):
        def method(self):
            return 'C' + super().method()

    class D(B, C):
        def method(self):
            return 'D' + super().method()

    d = D()
    result = d.method()  # Should be 'DBCA' due to MRO
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result = interpreter.get_name_value("result")
    assert result == "DBCA"


def test_complex_comprehensions():
    """Test complex comprehension scenarios."""
    code = """
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # Flatten matrix using nested comprehension
    flattened = [x for row in matrix for x in row]

    # Filter and transform in one comprehension
    result1 = [x * 2 for x in flattened if x % 2 == 0]

    # Dictionary comprehension with conditional
    result2 = {x: x**2 for x in range(5) if x % 2 == 0}
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result1 = interpreter.get_name_value("result1")
    result2 = interpreter.get_name_value("result2")
    assert result1 == [4, 8, 12, 16]
    assert result2 == {0: 0, 2: 4, 4: 16}


def test_complex_exception_handling():
    """Test complex exception handling scenarios."""
    code = """
    def risky_operation(x):
        if x < 0:
            raise ValueError("Negative value")
        if x == 0:
            raise TypeError("Zero value")
        return 1/x

    results = []
    for x in [2, 0, -1]:
        try:
            results.append(risky_operation(x))
        except ValueError as e:
            results.append(str(e))
        except TypeError as e:
            results.append(str(e))
        except ZeroDivisionError:
            results.append("Division by zero")
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    results = interpreter.get_name_value("results")
    assert results == [0.5, "Zero value", "Negative value"]


def test_complex_generator_expressions():
    """Test complex generator expression scenarios."""
    code = """
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    # Generator expression with condition
    result1 = list(x for x in range(10) if is_prime(x))

    # Nested generator expression
    matrix = [[1, 2], [3, 4], [5, 6]]
    result2 = list(x for row in matrix for x in row)

    # Generator expression with function call
    result3 = list(x * x for x in range(5))
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result1 = interpreter.get_name_value("result1")
    result2 = interpreter.get_name_value("result2")
    result3 = interpreter.get_name_value("result3")
    assert result1 == [2, 3, 5, 7]
    assert result2 == [1, 2, 3, 4, 5, 6]
    assert result3 == [0, 1, 4, 9, 16]


def test_complex_context_manager():
    """Test complex context manager scenarios."""
    code = """
    class Context:
        def __init__(self, name):
            self.name = name
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            return False

    results = []
    ctx1 = Context("test1")
    with ctx1:
        results.append(ctx1.entered)
        ctx2 = Context("test2")
        with ctx2:
            results.append(ctx2.entered)
        results.append(ctx2.exited)
    results.append(ctx1.exited)

    # Test context manager with exception
    ctx3 = Context("test3")
    try:
        with ctx3:
            results.append(ctx3.entered)
            raise ValueError("test error")
    except ValueError:
        results.append(ctx3.exited)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    results = interpreter.get_name_value("results")
    assert results == [True, True, True, True, True, True]


def test_complex_f_strings():
    """Test complex f-string formatting."""
    code = """
    x = 42
    y = "world"
    result1 = f"Value is {x:03d}"  # With format spec
    result2 = f"Hello {y!r}"  # With conversion
    result3 = f"{x = }"  # Debug syntax
    result4 = f"{x:>5}"  # With alignment
    result5 = f"{x:+}"  # With sign
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result1 = interpreter.get_name_value("result1")
    result2 = interpreter.get_name_value("result2")
    result3 = interpreter.get_name_value("result3")
    result4 = interpreter.get_name_value("result4")
    result5 = interpreter.get_name_value("result5")

    assert result1 == "Value is 042"
    assert result2 == "Hello 'world'"
    assert result3 == "x = 42"
    assert result4 == "   42"
    assert result5 == "+42"


def test_complex_f_string_errors():
    """Test error handling in f-strings."""
    code = """
    try:
        result = f"{1/0}"  # Should raise ZeroDivisionError
    except ZeroDivisionError as e:
        error1 = str(e)

    try:
        result = f"{x}"  # Should raise NameError
    except NameError as e:
        error2 = str(e)

    try:
        result = f"{42:invalid}"  # Should raise ValueError
    except ValueError as e:
        error3 = str(e)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    error1 = interpreter.get_name_value("error1")
    error2 = interpreter.get_name_value("error2")
    error3 = interpreter.get_name_value("error3")

    assert "division by zero" in error1
    assert "Name 'x' is not defined" in error2
    assert "Invalid format specification" in error3


def test_complex_subscript_operations():
    """Test complex subscript operations."""
    code = """
    lst = [1, 2, 3, 4, 5]
    result1 = lst[1:4:2]  # Slice with step
    result2 = lst[::-1]  # Reverse slice
    result3 = lst[...]  # Ellipsis

    matrix = [[1, 2], [3, 4]]
    result4 = matrix[1][0]  # Nested subscript

    d = {"a": 1, "b": 2}
    result5 = d["a"]  # Dictionary subscript
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    result1 = interpreter.get_name_value("result1")
    result2 = interpreter.get_name_value("result2")
    result3 = interpreter.get_name_value("result3")
    result4 = interpreter.get_name_value("result4")
    result5 = interpreter.get_name_value("result5")

    assert result1 == [2, 4]
    assert result2 == [5, 4, 3, 2, 1]
    assert result3 == [1, 2, 3, 4, 5]
    assert result4 == 3
    assert result5 == 1


def test_complex_subscript_errors():
    """Test error handling in subscript operations."""
    code = """
    lst = [1, 2, 3]

    try:
        result = lst[3]  # Should raise IndexError
    except IndexError as e:
        error1 = str(e)

    try:
        result = lst["key"]  # Should raise TypeError
    except TypeError as e:
        error2 = str(e)

    d = {"a": 1}
    try:
        result = d["b"]  # Should raise KeyError
    except KeyError as e:
        error3 = str(e)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    error1 = interpreter.get_name_value("error1")
    error2 = interpreter.get_name_value("error2")
    error3 = interpreter.get_name_value("error3")

    assert "list index out of range" in error1
    assert "list indices must be integers" in error2
    assert "'b'" in error3


def test_complex_attribute_errors():
    """Test error handling in attribute operations."""
    code = """
    class Test:
        pass

    t = Test()

    try:
        result = t.non_existent  # Should raise AttributeError
    except AttributeError as e:
        error1 = str(e)

    try:
        result = Test.non_existent  # Should raise AttributeError
    except AttributeError as e:
        error2 = str(e)
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    interpreter.execute(parser.parse(code))

    error1 = interpreter.get_name_value("error1")
    error2 = interpreter.get_name_value("error2")

    assert "'Test' object has no attribute 'non_existent'" in error1
    assert "type object 'Test' has no attribute 'non_existent'" in error2
