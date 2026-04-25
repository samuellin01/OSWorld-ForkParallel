# pyre-strict

import re
from contextlib import contextmanager

from typing import Any, Generator, get_args, get_origin

from pydantic import BaseModel, ValidationError


@contextmanager
def sanitize_pydantic_validation_error() -> Generator[None, Any, Any]:
    """
    Pydantic 2+ adds unhelpful suffix to the error message, we remove them
    similar to how we handle LC output parser error.
    It is unfortunate that we have to upcast it to ValueError, but it is not
    easy to re-create a ValidationError by ourselves.
    """

    try:
        yield
    except ValidationError as e:
        raise ValueError(
            re.sub(
                r"\s+For further information visit https:\/\/errors\.pydantic\.dev\/[0-9\.]+\/v\/\w+",
                "",
                str(e),
            )
        )


# Resembles get_args, but works for pydantic models as well
def cf_get_args(cls: Any) -> Any:
    if issubclass(cls, BaseModel):
        return cls.__pydantic_generic_metadata__["args"]
    return get_args(cls)


# Resembles get_origin, but works for pydantic models as well
def cf_get_origin(cls: Any) -> Any:
    if issubclass(cls, BaseModel):
        return cls.__pydantic_generic_metadata__["origin"]
    return get_origin(cls)
