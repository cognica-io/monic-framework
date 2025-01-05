#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import pytest

from monic.expressions import ExpressionsParser, ExpressionsInterpreter


def test_generator_yield():
    code = """
def gen():
    yield 1
    yield 2
    yield 3

result = list(gen())
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result") == [1, 2, 3]


def test_generator_yield_from():
    code = """
def subgen():
    yield 'a'
    yield 'b'
    return 'done'

def gen():
    print("script: GEN")
    output = yield from subgen()
    print("script: OUTPUT", output)
    yield output

print(gen)
result = list(gen())
print(result)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result") == ["a", "b", "done"]


def test_generator_yield_from_list():
    code = """
def gen():
    yield from [1, 2, 3]
    yield from 'abc'

result = list(gen())
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result") == [1, 2, 3, "a", "b", "c"]


def test_generator_yield_from_error():
    code = """
def gen():
    yield from 42
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)

    with pytest.raises(
        TypeError, match="cannot 'yield from' a non-iterator of type int"
    ):
        list(interpreter.execute(tree)())


def test_generator_with_return():
    code = """
def gen():
    yield 1
    yield 2
    return 'end'

def caller():
    generated = yield from gen()
    yield generated

result = list(caller())
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result") == [1, 2, "end"]


def test_generator_with_closure():
    code = """
def make_gen():
    x = 1
    def gen():
        nonlocal x
        yield x
        x += 1
        yield x
    return gen

g = make_gen()
print(g)
result1 = list(g())
result2 = list(g())
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result1") == [1, 2]
    assert interpreter.get_name_value("result2") == [2, 3]
