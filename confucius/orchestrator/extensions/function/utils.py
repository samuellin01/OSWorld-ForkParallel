# pyre-strict

import inspect
import json
import re
import types
from dataclasses import fields as dc_fields, is_dataclass
from functools import wraps

from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    get_args,
    get_origin,
    get_type_hints,
    List,
    Literal,
    Type,
    TypeVar,
    Union,
)

from langchain_core.runnables import RunnableLambda

from langchain_core.runnables.utils import is_async_callable, is_async_generator

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict

T = TypeVar("T")
Function = Callable[..., Awaitable[T] | T]

# Type alias for Union types created with the | operator
UnionTypeAlias = types.UnionType  # For Python 3.10+

# Type alias for primitive types in JSON schema
PythonPrimitiveType = Type[str] | Type[int] | Type[float] | Type[bool]


def get_runnable(func: Function, name: str | None = None) -> RunnableLambda:
    """
    Convert a function to a RunnableLambda.

    Args:
        func: The function to convert
        name: The name of the RunnableLambda. If not provided, it will be inferred from the function name.

    Returns:
        A RunnableLambda object
    """
    return RunnableLambda(get_single_kwargs_function(func), name=name)


def _convert_kwargs_to_typed_params(
    kwargs: Dict[str, Any], signature: inspect.Signature
) -> Dict[str, Any]:
    """
    Convert dictionary arguments to their corresponding types based on function parameter annotations.

    This handles conversion for pydantic models, dataclasses, and other complex types.

    Args:
        kwargs: The dictionary of arguments
        signature: The function signature with parameter type information

    Returns:
        The converted arguments dictionary
    """
    converted_kwargs = {}
    for name, param in signature.parameters.items():
        if name in kwargs:
            # Skip 'self' parameter for methods
            if name == "self":
                converted_kwargs[name] = kwargs[name]
                continue

            # Get parameter type annotation
            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )
            param_value = kwargs[name]

            # Only attempt conversion if we have a dict value and a valid type annotation
            if isinstance(param_value, dict):
                # Handle pydantic BaseModel
                if isinstance(param_type, type) and issubclass(param_type, BaseModel):
                    converted_kwargs[name] = param_type(**param_value)

                # Handle dataclasses - filter out fields that don't exist in the dataclass
                elif isinstance(param_type, type) and is_dataclass(param_type):
                    # Get all field names defined in the dataclass
                    field_names = {f.name for f in dc_fields(param_type)}
                    # Only include keys that exist in the dataclass
                    filtered_param_value = {
                        k: v for k, v in param_value.items() if k in field_names
                    }
                    converted_kwargs[name] = param_type(**filtered_param_value)

                # For TypedDict, we can use the dict directly (it's just a dict with type hints)
                # For other types, use as is
                else:
                    converted_kwargs[name] = param_value
            else:
                converted_kwargs[name] = param_value
        # If parameter not provided in kwargs, leave it out (will use default or raise error)

    return converted_kwargs


def get_single_kwargs_function(func: Function) -> Function:
    """
    Convert a function that takes multiple keyword arguments to a function that takes a single dictionary of keyword arguments.

    This function also handles automatic conversion of dictionary values to expected complex types such as:
    - pydantic models: If a parameter is annotated with a pydantic BaseModel subclass, dictionary values will be converted
    - dataclasses: If a parameter is annotated with a dataclass, dictionary values will be converted
    - TypedDict: If a parameter is annotated with a TypedDict, no conversion needed (it's already a dict)

    Args:
        func: The function to convert

    Returns:
        A new function that takes a single dictionary of keyword arguments
    """
    # Get function signature outside of the wrapper to avoid capturing it
    func_sig = inspect.signature(func)

    if is_async_generator(func):

        @wraps(func)
        # pyre-ignore: Missing return annotation [3]: Return type must be specified as type other than `Any`.
        async def async_gen_wrapper(kwargs: Dict[str, Any]) -> AsyncIterator[Any]:
            if not isinstance(kwargs, dict):
                raise TypeError("Argument must be a dictionary")

            # Convert arguments based on type annotations
            converted_kwargs = _convert_kwargs_to_typed_params(kwargs, func_sig)

            # Call the function with converted arguments
            async for item in func(**converted_kwargs):
                yield item

        return async_gen_wrapper
    elif is_async_callable(func):

        @wraps(func)
        # pyre-ignore: Missing return annotation [3]: Return type must be specified as type other than `Any`.
        async def async_wrapper(kwargs: Dict[str, Any]) -> Any:
            if not isinstance(kwargs, dict):
                raise TypeError("Argument must be a dictionary")

            # Convert arguments based on type annotations
            converted_kwargs = _convert_kwargs_to_typed_params(kwargs, func_sig)

            # Call the function with converted arguments
            return await func(**converted_kwargs)

        return async_wrapper
    else:

        @wraps(func)
        # pyre-ignore: Missing return annotation [3]: Return type must be specified as type other than `Any`.
        def sync_wrapper(kwargs: Dict[str, Any]) -> Any:
            if not isinstance(kwargs, dict):
                raise TypeError("Argument must be a dictionary")

            # Convert arguments based on type annotations
            converted_kwargs = _convert_kwargs_to_typed_params(kwargs, func_sig)

            # Call the function with converted arguments
            return func(**converted_kwargs)

        return sync_wrapper


AllTypes = (
    PythonPrimitiveType
    | UnionTypeAlias
    | Type[List[Any]]
    | Type[Dict[str, Any]]
    | Type[object]  # For custom types
)


def _handle_primitive_type(type_hint: AllTypes) -> Dict[str, Any] | None:
    """
    Handle primitive types in JSON schema conversion.

    Args:
        type_hint: Python primitive type

    Returns:
        JSON schema for the primitive type or None if not a primitive type
    """
    if type_hint is str:
        return {"type": "string"}
    elif type_hint is int:
        return {"type": "integer"}
    elif type_hint is float:
        return {"type": "number"}
    elif type_hint is bool:
        return {"type": "boolean"}
    elif type_hint is None or type_hint is type(None):
        return {"type": "null"}
    return None


def _handle_union_type(type_hint: AllTypes) -> Dict[str, Any] | None:
    """
    Handle Union types in JSON schema conversion.

    Args:
        type_hint: Python Union type

    Returns:
        JSON schema for the Union type or None if not a Union type
    """
    origin = get_origin(type_hint)
    if isinstance(type_hint, UnionTypeAlias) or origin is Union:
        args = get_args(type_hint)
        if type(None) in args:
            # If this is an Optional type (T | None), get the non-None type
            args = [arg for arg in args if arg is not type(None)]
        if len(args) == 1:
            return type_to_json_schema(args[0])

        # Handle union of multiple types
        any_of = [type_to_json_schema(t) for t in args]
        return {"anyOf": any_of}
    return None


def _handle_list_type(type_hint: AllTypes) -> Dict[str, Any] | None:
    """
    Handle List types in JSON schema conversion.

    Args:
        type_hint: Python List type

    Returns:
        JSON schema for the List type or None if not a List type
    """
    origin = get_origin(type_hint)
    if origin is list or type_hint is list:
        # If generic type is provided, use it; otherwise, default to any
        item_type = get_args(type_hint)
        item_schema = (
            type_to_json_schema(item_type[0]) if item_type else {"type": "object"}
        )
        return {"type": "array", "items": item_schema}
    return None


def _handle_dict_type(type_hint: AllTypes) -> Dict[str, Any] | None:
    """
    Handle Dict types in JSON schema conversion.

    Args:
        type_hint: Python Dict type

    Returns:
        JSON schema for the Dict type or None if not a Dict type
    """
    origin = get_origin(type_hint)
    if origin is dict or type_hint is dict:
        # Maintain compatibility with existing tests by always returning simple object schema
        return {"type": "object"}
    return None


def _handle_literal_type(type_hint: AllTypes) -> Dict[str, Any] | None:
    """
    Handle Literal types in JSON schema conversion.

    Args:
        type_hint: Python Literal type

    Returns:
        JSON schema for the Literal type or None if not a Literal type
    """
    origin = get_origin(type_hint)
    if origin is Literal:
        literal_values = get_args(type_hint)
        # Ensure all values are of the same type
        if literal_values and all(
            isinstance(val, type(literal_values[0])) for val in literal_values
        ):
            base_schema = type_to_json_schema(type(literal_values[0]))
            base_schema["enum"] = list(literal_values)
            return base_schema
        return {"enum": list(literal_values)}
    return None


def _handle_pydantic_model(type_hint: AllTypes) -> Dict[str, Any] | None:
    """
    Handle pydantic BaseModel types in JSON schema conversion.

    Args:
        type_hint: Python pydantic BaseModel type

    Returns:
        JSON schema for the pydantic BaseModel or None if not a pydantic BaseModel
    """
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        # For pydantic models, we can directly use the schema() method
        return type_hint.schema()
    return None


def _handle_typed_dict(type_hint: AllTypes) -> Dict[str, Any] | None:
    """
    Handle TypedDict types in JSON schema conversion.

    Args:
        type_hint: Python TypedDict type

    Returns:
        JSON schema for the TypedDict type or None if not a TypedDict
    """
    if isinstance(type_hint, type) and hasattr(type_hint, "__annotations__"):
        if isinstance(getattr(type_hint, "__total__", None), bool):  # TypedDict marker
            properties = {}
            required = []

            for field_name, field_type in type_hint.__annotations__.items():
                properties[field_name] = type_to_json_schema(field_type)

                # Check if this field is required (all fields are required by default unless using NotRequired)
                if getattr(type_hint, "__total__", True) or field_name in getattr(
                    type_hint, "__required_keys__", {field_name}
                ):
                    required.append(field_name)

            return {"type": "object", "properties": properties, "required": required}
    return None


def _handle_dataclass(type_hint: AllTypes) -> Dict[str, Any] | None:
    """
    Handle dataclass types in JSON schema conversion.

    Args:
        type_hint: Python dataclass type

    Returns:
        JSON schema for the dataclass type or None if not a dataclass
    """
    if isinstance(type_hint, type) and is_dataclass(type_hint):
        properties = {}
        required = []

        for field in dc_fields(type_hint):
            properties[field.name] = type_to_json_schema(field.type)

            # For dataclasses, fields without a default are required
            # In dataclasses, fields without default have default=dataclasses.MISSING
            # inspect.Parameter.empty is a placeholder we're using for comparison
            if field.default == field.default_factory:  # Both are default values
                # If both are equal, it means neither has been set (both are MISSING)
                required.append(field.name)

        return {"type": "object", "properties": properties, "required": required}
    return None


def type_to_json_schema(type_hint: AllTypes) -> Dict[str, Any]:
    """
    Convert Python type hints to JSON Schema representation.

    Supports primitive types, Union types (both Union[T, None] and T | None syntax),
    Lists, Dictionaries, pydantic BaseModel, TypedDict, dataclasses, and gracefully
    handles other complex types.

    Args:
        type_hint: Python type annotation to convert to JSON schema

    Returns:
        Dictionary representing JSON Schema for the type

    Examples:
        >>> type_to_json_schema(str)
        {'type': 'string'}
        >>> type_to_json_schema(Optional[int])  # or int | None in Python 3.10+
        {'type': 'integer'}
        >>> type_to_json_schema(List[str])
        {'type': 'array', 'items': {'type': 'string'}}
        >>> # For a pydantic model, returns the result of model.schema()
        >>> # For TypedDict and dataclasses, generates appropriate schema
    """
    # Try each handler in sequence
    for handler in [
        _handle_primitive_type,
        _handle_union_type,
        _handle_list_type,
        _handle_dict_type,
        _handle_literal_type,
        _handle_pydantic_model,
        _handle_typed_dict,
        _handle_dataclass,
    ]:
        result = handler(type_hint)
        if result is not None:
            return result

    # Default to object for complex types or unrecognized types
    return {"type": "object"}


def _enhance_schema_with_metadata(schema: dict[str, Any], func: Function) -> None:
    """
    Enhance schema properties with docstring descriptions and default values.

    Args:
        schema: Base schema to enhance (modified in-place)
        func: Function to extract metadata from
    """
    if "properties" not in schema:
        return

    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if name == "self" or name not in schema["properties"]:
            continue

        # Add docstring description if available
        if func.__doc__:
            param_desc_match = re.search(
                rf"\s+{name}:\s*(.*?)(?:\n\s+\w+:|$)",
                func.__doc__,
                re.MULTILINE | re.DOTALL,
            )
            if param_desc_match:
                param_desc = param_desc_match.group(1).strip()
                schema["properties"][name]["description"] = param_desc

        # Add default value if present
        if param.default != inspect.Parameter.empty and param.default is not None:
            schema["properties"][name]["default"] = param.default


def _generate_unified_schema_with_typeadapter(func: Function) -> dict[str, Any]:
    """
    Generate unified schema using TypeAdapter with smart substitution.

    Uses unified TypedDict approach to generate proper $defs sections for Pydantic models
    while substituting incompatible types (like pandas.DataFrame) with dict to maintain
    unified schema generation.

    Args:
        func: Function to generate schema for

    Returns:
        dict: JSON schema with proper $defs section and graceful fallback for incompatible types
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Build unified type representing all function parameters
    field_definitions = {}
    required_fields = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        param_type = type_hints.get(name, Any)

        try:
            TypeAdapter(param_type)
            field_definitions[name] = param_type
        except Exception:
            # Use dict as substitute - generates proper object schema with additionalProperties
            field_definitions[name] = dict

        if param.default == inspect.Parameter.empty:
            required_fields.append(name)

    # Create TypedDict and generate schema using TypeAdapter
    # pyre-ignore: TypedDict callable annotation issue
    FunctionArgsType = TypedDict(f"{func.__name__}Args", field_definitions)
    adapter = TypeAdapter(FunctionArgsType)
    schema = adapter.json_schema()

    # Add required fields (TypedDict doesn't preserve this from function signature)
    schema["required"] = required_fields

    # Enhance schema with docstring descriptions and default values
    _enhance_schema_with_metadata(schema, func)

    return schema


def _generate_schema_original_method(func: Function) -> dict[str, Any]:
    """Original implementation - kept as fallback in case TypeAdapter approach fails."""
    # Get function signature
    sig = inspect.signature(func)

    # Prepare schema structure
    schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # Process each parameter
    for name, param in sig.parameters.items():
        # Skip 'self' parameter for methods
        if name == "self" and len(sig.parameters) > 0:
            continue

        # Determine type (use Any if no annotation)
        type_hint = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any
        )

        # Generate schema for the parameter
        param_schema = type_to_json_schema(type_hint)

        # Add parameter description from docstring if available
        if func.__doc__:
            param_desc_match = re.search(
                rf"\s+{name}:\s*(.*?)(?:\n\s+\w+:|$)",
                func.__doc__,
                re.MULTILINE | re.DOTALL,
            )
            if param_desc_match:
                param_desc = param_desc_match.group(1).strip()
                param_schema["description"] = param_desc

        # Add default value to schema if provided
        if param.default != inspect.Parameter.empty and param.default is not None:
            param_schema["default"] = param.default

        # Add to properties
        # pyre-ignore: Undefined attribute [16]: Item `str` of `typing.Union[Dict[typing.Any, typing.Any], typing.List[typing.Any], str]` has no attribute `__setitem__`.
        schema["properties"][name] = param_schema

        # Add to required if no default value
        if param.default == inspect.Parameter.empty:
            # pyre-ignore: Undefined attribute [16]: Item `Dict` of `typing.Union[Dict[typing.Any, typing.Any], typing.List[typing.Any], str]` has no attribute `append`.
            schema["required"].append(name)

    return schema


def generate_function_json_schema_dict(func: Function) -> dict[str, Any]:
    """
    Generate JSON schema for a function's parameters.

    This function inspects the signature of the provided function and generates
    a JSON schema that describes its parameters, including their types and whether
    they are required.

    Args:
        func: The function to generate schema for

    Returns:
        JSON schema as a dictionary with properties and required fields

    Examples:
        >>> def example_func(name: str, age: int = 30, tags: List[str] = None):
        ...     pass
        >>> schema = generate_function_json_schema_dict(example_func)
        >>> schema["properties"]["name"]["type"]
        'string'
        >>> schema["properties"]["age"]["type"]
        'integer'
        >>> "name" in schema["required"]
        True
        >>> "age" in schema["required"]
        False
    """
    use_typeadapter = True

    if use_typeadapter:
        # Use new TypeAdapter approach
        return _generate_unified_schema_with_typeadapter(func)
    else:
        # Use original method (kept for backward compatibility)
        return _generate_schema_original_method(func)


def generate_function_json_schema(func: Function, **kwargs: Any) -> str:
    """
    Generate JSON schema for a function's parameters.

    Args:
        func: The function to generate schema for
        kwargs: Additional arguments to pass to json.dumps

    Returns:
        JSON schema as a string
    """

    return json.dumps(generate_function_json_schema_dict(func), **kwargs)
