# pyre-strict

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .llm_params import LLMParams

_llm_params: ContextVar[LLMParams] = ContextVar("params", default=LLMParams())


class LLMManager(BaseModel):
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @contextmanager
    # pyre-fixme[3]: Return type must be annotated.
    def override(self, **kwargs: Any):
        old_value = _llm_params.get()
        try:
            updates = LLMParams.parse_obj(kwargs)
            _llm_params.set(
                old_value.model_copy(update=updates.dict(exclude_defaults=True))
            )
            yield self
        finally:
            _llm_params.set(old_value)

    @property
    def llm_params(self) -> LLMParams:
        return _llm_params.get()

    def _get_llm_params(self, **kwargs: Any) -> LLMParams:
        return LLMParams(**kwargs).model_copy(
            update=_llm_params.get().dict(exclude_defaults=True)
        )

    def get_chat(self, **kwargs: Any) -> BaseChatModel:
        return self._get_chat(
            params=self._get_llm_params(**kwargs),
        )

    def _get_chat(self, params: LLMParams) -> BaseChatModel:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support chat model."
        )

    def get_embeddings(self, **kwargs: Any) -> Embeddings:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings."
        )
