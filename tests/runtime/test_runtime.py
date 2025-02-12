#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import datetime

import pytest

from monic.expressions import ExpressionsParser, ExpressionsInterpreter


def _has_module(module_name: str) -> bool:
    code = f"""
    {module_name}.is_available()
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    return interpreter.execute(tree)


@pytest.mark.skipif(not _has_module("json"), reason="JSON is not available")
def test_json_runtime():
    code = """
# Test json.dumps
json_str = json.dumps({"a": 1, "b": [2, 3]})

# Test json.loads
parsed_obj = json.loads(json_str)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert interpreter.get_name_value("json_str") == '{"a": 1, "b": [2, 3]}'
    assert interpreter.get_name_value("parsed_obj") == {"a": 1, "b": [2, 3]}


@pytest.mark.skipif(not _has_module("time"), reason="Time is not available")
def test_time_runtime():
    code = """
# Test time.time() and time.monotonic()
current_time = time.time()
monotonic_time = time.monotonic()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.get_name_value("current_time"), float)
    assert isinstance(interpreter.get_name_value("monotonic_time"), float)


@pytest.mark.skipif(
    not _has_module("datetime"), reason="Datetime is not available"
)
def test_datetime_runtime():
    code = """
# Test datetime module binding
current_date = datetime.date.today()
current_datetime = datetime.datetime.now()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.get_name_value("current_date"), datetime.date)
    assert isinstance(
        interpreter.get_name_value("current_datetime"), datetime.datetime
    )


@pytest.mark.skipif(not _has_module("np"), reason="Numpy is not available")
def test_numpy_runtime():
    code = """
    arr = np.array([1, 2, 3])
    mean_val = np.mean(arr)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.get_name_value("arr"), object)
    assert isinstance(interpreter.get_name_value("mean_val"), (int, float))


@pytest.mark.skipif(not _has_module("pd"), reason="Pandas is not available")
def test_pandas_runtime():
    code = """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    mean_val = df["a"].mean()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.get_name_value("df"), object)
    assert isinstance(interpreter.get_name_value("mean_val"), (int, float))


@pytest.mark.skipif(not _has_module("pl"), reason="Polars is not available")
def test_polars_runtime():
    code = """
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    mean_val = df["a"].mean()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.get_name_value("df"), object)
    assert isinstance(interpreter.get_name_value("mean_val"), (int, float))


@pytest.mark.skipif(not _has_module("pa"), reason="Arrow is not available")
def test_arrow_runtime():
    code = """
    table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.get_name_value("table"), object)


@pytest.mark.skipif(
    not _has_module("pc"), reason="Arrow Compute is not available"
)
def test_arrow_compute_runtime():
    code = """
    table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    mean_val = pc.mean(table["a"]).as_py()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.get_name_value("table"), object)
    assert isinstance(interpreter.get_name_value("mean_val"), (int, float))


@pytest.mark.skipif(
    not _has_module("pq"), reason="Arrow Parquet is not available"
)
def test_arrow_parquet_runtime():
    code = """
    result = pq.is_available()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.get_name_value("result"), bool)


@pytest.mark.skipif(
    not _has_module("inspector"), reason="Inspector is not available"
)
def test_inspector_is_available():
    code = """
    result = inspector.is_available()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.get_name_value("result"), bool)


@pytest.mark.skipif(
    not _has_module("inspector"), reason="Inspector is not available"
)
def test_inspector_runtime():
    code = """
    sig = inspector.signature(json.dumps)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    sig = interpreter.get_name_value("sig")
    assert isinstance(sig, str)
    assert "json.dumps" in sig

    code = """
    def add(a: int, b: int) -> int:
        return a + b

    sig = inspector.signature(add)
"""
    tree = parser.parse(code)
    interpreter.execute(tree)

    sig = interpreter.get_name_value("sig")
    assert isinstance(sig, str)
    assert "add" in sig

    code = """
    sig = inspector.signature(json)
"""
    with pytest.raises(ValueError):
        tree = parser.parse(code)
        interpreter.execute(tree)
