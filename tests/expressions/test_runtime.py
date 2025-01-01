#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import datetime

from monic.expressions import ExpressionsParser, ExpressionsInterpreter


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

    assert interpreter.local_env["json_str"] == '{"a": 1, "b": [2, 3]}'
    assert interpreter.local_env["parsed_obj"] == {"a": 1, "b": [2, 3]}


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

    assert isinstance(interpreter.local_env["current_time"], float)
    assert isinstance(interpreter.local_env["monotonic_time"], float)


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

    assert isinstance(interpreter.local_env["current_date"], datetime.date)
    assert isinstance(
        interpreter.local_env["current_datetime"], datetime.datetime
    )


def test_numpy_runtime():
    code = """
arr = np.array([1, 2, 3])
mean_val = np.mean(arr)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.local_env["arr"], object)
    assert isinstance(interpreter.local_env["mean_val"], (int, float))


def test_pandas_runtime():
    code = """
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
mean_val = df["a"].mean()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.local_env["df"], object)
    assert isinstance(interpreter.local_env["mean_val"], (int, float))


def test_polars_runtime():
    code = """
df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
mean_val = df["a"].mean()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)

    assert isinstance(interpreter.local_env["df"], object)
    assert isinstance(interpreter.local_env["mean_val"], (int, float))
