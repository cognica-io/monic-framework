#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

# pylint: disable=redefined-builtin

from fastapi import FastAPI, Query
from pydantic import BaseModel

from monic.expressions import (
    ExpressionsContext,
    ExpressionsParser,
    ExpressionsInterpreter,
    monic_bind,
)


app = FastAPI(title="Compute API")


class ComputeInput(BaseModel):
    input: str
    timeout: float = 10.0


@monic_bind
def say_hello(name: str) -> str:
    return f"Hello, {name}!"


@app.get("/compute")
async def compute_get(
    input: str = Query(..., description="Expression to compute"),
    timeout: float = Query(10.0, description="Timeout for computation"),
):
    """
    Compute endpoint supporting GET method
    Returns the computed result of the input expression
    """
    parser = ExpressionsParser()
    context = ExpressionsContext(timeout=timeout)
    interpreter = ExpressionsInterpreter(context=context)
    result = interpreter.execute(parser.parse(input))

    return {"result": result}


@app.post("/compute")
async def compute_post(compute_input: ComputeInput):
    """
    Compute endpoint supporting POST method
    Returns the computed result of the input expression
    """
    parser = ExpressionsParser()
    context = ExpressionsContext(timeout=compute_input.timeout)
    interpreter = ExpressionsInterpreter(context=context)
    result = interpreter.execute(parser.parse(compute_input.input))

    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
