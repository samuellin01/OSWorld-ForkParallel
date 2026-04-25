# pyre-strict
from inspect import isclass
from types import get_original_bases
from typing import Any, Generic, List, Optional, Tuple, Type, TypeVar

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from ...utils.pydantic import cf_get_args, cf_get_origin


Input = TypeVar("Input", bound=BaseModel)
Output = TypeVar("Output", bound=BaseModel)


class AnalectBase(BaseModel, Runnable[Input, Output], Generic[Input, Output]):
    """Base class for Analects that combines Pydantic models with LangChain runnables."""

    name: Optional[str] = Field(
        default=None,
        description="Custom runnable name for the Analect",
    )

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        return Runnable.get_name(
            self, suffix, name=self.name or self.__class__.__name__
        )

    def get_description(self) -> str:
        """
        Description for the analect
        """
        return (self.__doc__ or "").strip()

    @classmethod
    def get_input_type(cls) -> Type[Input]:
        raise NotImplementedError()

    @classmethod
    def get_output_type(cls) -> Type[Output]:
        raise NotImplementedError()

    @classmethod
    def _get_input_output(cls) -> Tuple[Type[Input], Type[Output]]:
        stack = [cls]
        while stack:
            cls_or_alias = stack.pop()

            # Skip processing python generic alias. This is ok since in pydantic v2,
            # we do not use python standard generic alias to lookup runtime input/output type.
            if not isclass(cls_or_alias):
                continue

            klass = cls_or_alias  # now we know it is a class

            # no need to pursue non-analects
            if not issubclass(klass, AnalectBase):
                continue

            search_base = cls._input_output_type_search_base()
            if klass is search_base:
                raise RuntimeError(
                    "Unable to get input/output type information from unparameterized-Analect class. "
                    "This error usually happens when you are inheriting from a subclass of Analect "
                    "which either partially or not specialize the input/output type (e.g. LLMAnalect). "
                    "To solve this, make sure to override 'get_input_type' and 'get_output_type' to "
                    "explicitly specify the input/output type (preferred), or override xxx "
                    "(less preferred, only if you know what you are doing!)."
                )
            elif cf_get_origin(klass) is search_base:
                return cf_get_args(klass)
            else:
                for c in get_original_bases(klass):
                    stack.append(c)

        raise ValueError("No analect base found")

    @classmethod
    # This actually needs to be the unparameterized type
    # pyre-fixme[24]: Invalid type parameters [24]: Generic type `AnalectBase` expects 2 type parameters.
    def _input_output_type_search_base(cls) -> Type["AnalectBase"]:
        # expecting this type to be contain two type parameters,
        # corresponding to input and output of the Analect
        return AnalectBase

    @property
    def input_keys(self) -> List[str]:
        assert issubclass(
            self.get_input_type(), BaseModel
        ), f"Input must be a Pydantic BaseModel, but got {type(self.get_input_type())}"
        return list(self.get_input_type().__fields__)

    @property
    def output_keys(self) -> List[str]:
        assert issubclass(
            self.get_output_type(), BaseModel
        ), f"Output must be a Pydantic BaseModel, but got {type(self.get_output_type())}"
        return list(self.get_output_type().__fields__)

    @classmethod
    def input(cls, *args: Any, **kwargs: Any) -> Input:
        return cls.get_input_type()(*args, **kwargs)

    @classmethod
    def output(cls, *args: Any, **kwargs: Any) -> Output:
        return cls.get_output_type()(*args, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _chain_type(self) -> str:
        return type(self).__name__
