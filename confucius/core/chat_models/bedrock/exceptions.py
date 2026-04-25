# pyre-strict

import logging
from contextlib import contextmanager
from typing import Any, Generator


logger: logging.Logger = logging.getLogger(__name__)


class BedrockBaseException(Exception):
    """Base exception class for Bedrock exceptions with metadata support."""

    def __init__(
        self, message: str = "", metadata: dict[str, Any] | None = None
    ) -> None:
        self.metadata = metadata
        super().__init__(message)

    def __str__(self) -> str:
        base_str = super().__str__()
        if self.metadata:
            return f"{base_str} - Metadata: {self.metadata}"
        return base_str


class BedrockThrottlingException(BedrockBaseException):
    """Exception raised when Bedrock API throttles the request."""

    pass


class BedrockValidationException(BedrockBaseException):
    """Exception raised when Bedrock API validation fails."""

    pass


class BedrockServiceUnavailableException(BedrockBaseException):
    """Exception raised when Bedrock service is unavailable."""

    pass


class BedrockModelErrorException(BedrockBaseException):
    """Exception raised when an unexpected Bedrock exception is received."""

    pass


class UnexpectedEmptyResponseException(BedrockBaseException):
    """Exception raised when an empty response is received unexpectedly."""

    pass


class BedrockInvalidResponseException(BedrockBaseException):
    """Exception raised when an invalid response is received."""

    pass


class BedrockAccessDeniedException(BedrockBaseException):
    """Exception raised when access is denied to Bedrock resources."""

    pass


class BedrockResourceNotFoundException(BedrockBaseException):
    """Exception raised when a Bedrock resource is not found."""

    pass


class BedrockModelTimeoutException(BedrockBaseException):
    """Exception raised when a Bedrock model operation times out."""

    pass


class BedrockInternalServerException(BedrockBaseException):
    """Exception raised when Bedrock encounters an internal server error."""

    pass


class BedrockServiceQuotaExceededException(BedrockBaseException):
    """Exception raised when Bedrock service quota is exceeded."""

    pass


class BedrockModelNotReadyException(BedrockBaseException):
    """Exception raised when Bedrock model is not ready."""

    pass


# Mapping from boto3 exception names to our custom exception classes
BEDROCK_EXCEPTION_MAPPING: dict[str, type[BedrockBaseException]] = {
    "ValidationException": BedrockValidationException,
    "ServiceUnavailableException": BedrockServiceUnavailableException,
    "ThrottlingException": BedrockThrottlingException,
    "ModelErrorException": BedrockModelErrorException,
    "AccessDeniedException": BedrockAccessDeniedException,
    "ResourceNotFoundException": BedrockResourceNotFoundException,
    "ModelTimeoutException": BedrockModelTimeoutException,
    "InternalServerException": BedrockInternalServerException,
    "ServiceQuotaExceededException": BedrockServiceQuotaExceededException,
    "ModelNotReadyException": BedrockModelNotReadyException,
}


def _get_client_exception_metadata(exc: BaseException) -> dict[str, Any] | None:
    """Extract metadata from a boto3 client exception."""
    response = getattr(exc, "response", {})
    return response.get("ResponseMetadata") if response is not None else None


def _handle_bedrock_exception(exc: BaseException) -> None:
    """Convert boto3 bedrock exceptions to our custom exceptions."""
    exc_name = exc.__class__.__name__
    exc_message = str(exc)
    logger.error(f"Encountered bedrock exception: {exc_name}[{exc_message}]")
    if exc_name in BEDROCK_EXCEPTION_MAPPING:
        custom_exc_class = BEDROCK_EXCEPTION_MAPPING[exc_name]
        metadata = _get_client_exception_metadata(exc)
        raise custom_exc_class(exc_message, metadata=metadata) from exc
    else:
        # Fallback for unknown bedrock exceptions
        metadata = _get_client_exception_metadata(exc)
        raise BedrockBaseException(
            f"Unexpected bedrock error: {exc}", metadata=metadata
        ) from exc


@contextmanager
def bedrock_exception_handling() -> Generator[None, None, None]:
    """Context manager for handling bedrock exceptions."""
    try:
        yield
    except BaseException as e:  # noqa: B036 - We re-raise non-bedrock exceptions
        # Check if this is a bedrock client exception by examining the module path OR class name
        is_bedrock_exception = (
            # Check module path (for real boto3 exceptions)
            (
                hasattr(e, "__module__")
                and e.__module__
                and "botocore.exceptions" in e.__module__
            )
            or
            # Check class name (for test mocks and real exceptions)
            (e.__class__.__name__ in BEDROCK_EXCEPTION_MAPPING)
        )

        if is_bedrock_exception:
            _handle_bedrock_exception(e)
        else:
            # Re-raise non-bedrock exceptions
            raise
