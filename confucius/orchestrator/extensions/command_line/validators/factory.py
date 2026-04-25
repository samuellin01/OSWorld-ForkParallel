# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Dict, Optional, Set

from .awk_validator import AwkValidator
from .cli_command_validator import CliCommandValidator
from .sort_validator import SortValidator
from .uniq_validator import UniqValidator


def get_default_validators() -> Dict[str, CliCommandValidator]:
    """Get a dictionary of default command validators.

    This function returns a pre-configured set of validators for commonly used
    commands that need validation for security or usability reasons.

    Returns:
        Dictionary mapping command names to validator instances
    """
    validators: Dict[str, CliCommandValidator] = {
        "awk": AwkValidator(),
        "sort": SortValidator(),
        "uniq": UniqValidator(),
        # Add more validators here as they're implemented
    }

    return validators


def create_validator_registry(
    *,
    include_default: bool = True,
    custom_validators: Optional[Dict[str, CliCommandValidator]] = None,
    exclude: Optional[Set[str]] = None,
) -> Dict[str, CliCommandValidator]:
    """Create a validator registry with optional customization.

    Args:
        include_default: Whether to include the default validators
        custom_validators: Additional custom validators to include
        exclude: Command names to exclude from the registry

    Returns:
        Dictionary mapping command names to validator instances
    """
    # Start with default validators if requested
    validators: Dict[str, CliCommandValidator] = {}
    if include_default:
        validators = get_default_validators()

    # Add custom validators if provided
    if custom_validators:
        validators.update(custom_validators)

    # Remove excluded validators if specified
    if exclude:
        for command in exclude:
            validators.pop(command, None)

    return validators
