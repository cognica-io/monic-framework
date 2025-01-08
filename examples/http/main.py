#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

# pylint: disable=redefined-builtin,broad-exception-caught

import typing as t

from fastapi import FastAPI, Query
from pydantic import BaseModel

from monic.expressions import (
    ExpressionsContext,
    ExpressionsParser,
    ExpressionsInterpreter,
    monic_bind,
)


app = FastAPI(title="Compute API")


@monic_bind
def say_hello(name: str) -> str:
    return f"Hello, {name}!"


class ComputeInput(BaseModel):
    input: str
    timeout: float = 10.0


T = t.TypeVar("T")


class Pod(BaseModel, t.Generic[T]):
    type: str
    content: T | None


class ComputeResponse(BaseModel, t.Generic[T]):
    success: bool
    error: Pod[T] | None
    pods: list[Pod[T]]


def compute(input: str, timeout: float) -> ComputeResponse:
    try:
        parser = ExpressionsParser()
        context = ExpressionsContext(timeout=timeout)
        interpreter = ExpressionsInterpreter(context=context)
        result = interpreter.execute(parser.parse(input))
    except Exception as e:
        return ComputeResponse(
            success=False,
            error=Pod(type=type(e).__name__, content=str(e)),
            pods=[],
        )

    return ComputeResponse(
        success=True,
        error=None,
        pods=[Pod(type=type(result).__name__, content=result)],
    )


@app.get("/compute")
async def compute_get(
    input: str = Query(..., description="Expression to compute"),
    timeout: float = Query(10.0, description="Timeout for computation"),
) -> ComputeResponse:
    """
    Compute endpoint supporting GET method
    Returns the computed result of the input expression
    """
    return compute(input, timeout)


@app.post("/compute")
async def compute_post(compute_input: ComputeInput) -> ComputeResponse:
    """
    Compute endpoint supporting POST method
    Returns the computed result of the input expression
    """
    return compute(compute_input.input, compute_input.timeout)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
