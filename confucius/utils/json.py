# pyre-strict
import json
import re
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Pattern to match a backslash followed by any non-valid escape sequence
# pyre-fixme[5]: Global expression must be annotated.
INVALID_ESCAPE_PATTERN = re.compile(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


def serialize_object(
    obj: Any,
    include: List[str] | Callable[[Any, str], bool] | None = None,
    exclude: List[str] | Callable[[Any, str], bool] | None = None,
    recursive: bool = False,
) -> Any:
    """
    serialize an object by (recursively) including or excluding fields.

    Parameters
    ----------
    obj : Any
        The object to serialize.
    include : list of str, or a callable, optional
        If it is a list of str, then it is a list of keys to include in the output.
        If it is a callable, then it should take an object as input and return whether to include that object in the output.
        If not specified, all keys will be included.
    exclude : list of str, or a callable, optional
        If it is a list of str, then it is a list of keys to exclude in the output.
        If it is a callable, then it should take an object as input and return whether to exclude that object in the output.
        If not specified, no keys will be excluded.
    recursive : bool, default=False
        Whether to recursively include/exclude fields in nested objects.

    Returns
    -------
    Any
        The serialized object.
    """

    # pyre-fixme[3]: Return type must be annotated.
    def _serialize_object(_obj):
        if isinstance(_obj, dict):
            # If include is defined, only consider those keys
            if include:
                _obj = {
                    key: value
                    for key, value in _obj.items()
                    if (
                        (key in include)
                        if isinstance(include, list)
                        else include(_obj, key)
                    )
                }

            # Exclude the keys from exclude list
            if exclude:
                _obj = {
                    key: value
                    for key, value in _obj.items()
                    if (
                        (key not in exclude)
                        if isinstance(exclude, list)
                        else not exclude(_obj, key)
                    )
                }

            # If recursive, apply custom_serializer recursively on dict values
            if recursive:
                return {key: _serialize_object(value) for key, value in _obj.items()}

        elif isinstance(_obj, list) and recursive:
            return [_serialize_object(item) for item in _obj]

        return _obj

    return _serialize_object(obj)


def json_dumps(
    obj: Any,
    include: List[str] | Callable[[Any, str], bool] | None = None,
    exclude: List[str] | Callable[[Any, str], bool] | None = None,
    recursive: bool = False,
    **kwargs: Any,
) -> str:
    """
    Dump a JSON object to string, but with extended support for (recursively) include or exclude fields.

    Parameters
    ----------
    obj, include, exclude, recursive, kwargs : same as `serialize_object`
    kwargs : dict
        Additional keyword arguments passed to `json.dumps`.

    Returns
    -------
    str
        The JSON representation of the object.
    """
    return json.dumps(
        serialize_object(obj, include=include, exclude=exclude, recursive=recursive),
        **kwargs,
    )


def expand_json_schema_refs(schema: Any, root_schema: Any | None = None) -> Any:
    """
    Recursively expand $ref references in a JSON schema.

    Given a JSON schema that contains $ref references to other parts of the schema,
    this function will replace those references with the actual content they point to,
    resulting in a fully expanded schema without any $ref references.

    Parameters:
    ----------
    schema : dict
        The JSON schema to be expanded. This can be a full schema or a subsection of a schema.

    root_schema : dict, optional
        The root schema from which references might be made. This should be the top-level schema
        that contains all possible definitions that might be referenced. If not provided,
        it defaults to the `schema` itself.

    Returns:
    -------
    dict
        The expanded schema with all $ref references replaced by their actual content.
    """
    if root_schema is None:
        root_schema = schema

    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_path = schema["$ref"].split("/")[1:]
            ref_value = root_schema
            for path in ref_path:
                ref_value = ref_value[path]
            return expand_json_schema_refs(ref_value, root_schema=root_schema)
        else:
            # Handle "anyOf", "allOf", "oneOf" keywords for Union and other composite types
            for keyword in ["anyOf", "allOf", "oneOf"]:
                if keyword in schema:
                    schema[keyword] = [
                        expand_json_schema_refs(item, root_schema=root_schema)
                        for item in schema[keyword]
                    ]
            return {
                key: expand_json_schema_refs(value, root_schema=root_schema)
                for key, value in schema.items()
            }
    elif isinstance(schema, list):
        return [
            expand_json_schema_refs(item, root_schema=root_schema) for item in schema
        ]

    return schema


def _valid_decode_json_schema_include(obj: Any, key: str) -> bool:
    """
    Following https://fburl.com/code/26en079d
    We need to make sure we don't include any field that is not supported by MetaGen service
    """
    if not isinstance(obj, dict):
        return True

    if "type" not in obj:
        return True

    type_ = obj["type"]

    if type_ == "object":
        return key in ["type", "description", "properties", "required"]
    elif type_ == "array":
        return key in ["type", "description", "items"]
    elif type_ in ["string", "boolean", "number", "integer"]:
        return key in ["type", "description"]
    elif type_ == "null":
        return key in ["type"]
    else:
        raise ValueError(
            f"Unsupported type: {type_} for generating valid JSON Schema for guided decode"
        )


def _map_json_schema_types(schema: Any) -> Any:
    """
    Post-processes a JSON schema to replace some unsupported types to supported types.

    1. Replacing all instances of 'integer' and 'float' with 'number'.
    2. Replacing enum types to 'string'.
    3. Add empty list for 'object' if there is no 'required' field.

    Parameters:
    ----------
    schema : dict
        The JSON schema to be processed.

    Returns:
    -------
    dict
        The processed schema with all numerical types replaced by 'number'.
    """

    if isinstance(schema, dict):
        if "type" in schema:
            if schema["type"] in ["integer", "float"]:
                schema["type"] = "number"
            if schema["type"] == "object":
                if "required" not in schema:
                    schema["required"] = []
        if "enum" in schema:
            schema["type"] = "string"
        # Recursively process all dictionary items
        return {key: _map_json_schema_types(value) for key, value in schema.items()}

    elif isinstance(schema, list):
        # Recursively process all list items
        return [_map_json_schema_types(item) for item in schema]

    return schema


def get_valid_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a valid JSON schema for guided decode.

    Parameters:
    ----------
    schema : dict
        A JSON schema.

    Returns:
    -------
    dict
        A valid JSON schema for guided decode.
    """
    schema = _map_json_schema_types(schema)
    schema = expand_json_schema_refs(schema)
    return serialize_object(
        schema,
        include=_valid_decode_json_schema_include,
        recursive=True,
    )


def get_pydantic_schema(pydantic_type: Type[T]) -> Dict[str, Any]:
    """
    Get a JSON schema for a given Pydantic model.

    Parameters:
    ----------
    pydantic_type : Type[T]
        A Pydantic model class.

    Returns:
    -------
    dict
        A JSON schema for the given Pydantic model.
    """
    try:
        schema = pydantic_type.model_json_schema()
    except (ModuleNotFoundError, AttributeError):
        # some external code is relying on this pydantic v1 feature
        # this code path is untested and unmaintained
        schema = pydantic_type.schema()

    return get_valid_schema(schema)


def get_pydantic_schema_json(pydantic_type: Type[T], **kwargs: Any) -> str:
    """
    Same as get_pydantic_schema but returning a JSON string.
    """
    return json.dumps(get_pydantic_schema(pydantic_type), **kwargs)


def get_schema(data: Any) -> Any:
    """
    Generate a basic JSON Schema from a Python dictionary.

    Parameters:
    ----------
    data : Any
        The Python object from which to generate the JSON Schema.
        The object can be a dictionary, list, string, int, float, boolean, or a Pydantic model.

    Returns:
    -------
    Any
        A basic JSON Schema representing the provided dictionary.
    """
    if isinstance(data, dict):
        properties = {}
        for key, value in data.items():
            properties[key] = get_schema(value)

        return {
            "type": "object",
            "properties": properties,
            "required": list(data.keys()),
        }

    elif isinstance(data, list):
        if data:
            items_schema = get_schema(data[0])
        else:
            items_schema = {}
        return {"type": "array", "items": items_schema}

    elif isinstance(data, str):
        return {"type": "string"}

    elif isinstance(data, (int, float)):
        return {"type": "number"}

    elif isinstance(data, bool):
        return {"type": "boolean"}

    elif isinstance(data, Enum):
        return {"type": "string"}

    elif isinstance(data, BaseModel):
        return get_pydantic_schema(type(data))

    elif isinstance(data, type) and issubclass(data, BaseModel):
        return get_pydantic_schema(data)
    else:
        raise TypeError(f"Unable to infer schema for {str(data)}")


def get_schema_json(data: Any) -> str:
    """
    Same as get_schema but returning a JSON string.
    """
    return json.dumps(get_schema(data))


def remove_trailing_commas(json_string: str) -> str:
    json_string = re.sub(r",\s*}", "}", json_string)
    json_string = re.sub(r",\s*]", "]", json_string)
    return json_string


def fix_invalid_escapes(json_str: str) -> str:
    # Replace invalid escape sequences with their literal representations
    fixed_json_str = INVALID_ESCAPE_PATTERN.sub("", json_str)
    return fixed_json_str


def _extract_jsons_impl(
    text: str,
) -> Tuple[List[Dict[str, Any]], List[json.JSONDecodeError]]:
    objs = []
    errs = []

    start = 0
    while start < len(text):
        if text[start] != "{":
            start += 1
            continue

        end = start
        stack = []
        prev_num_validations = len(objs) + len(errs)
        while end < len(text):
            if text[end] == "{":
                stack.append(text[end])
            elif text[end] == "}":
                if not stack:
                    break
                stack.pop()
                if not stack:  # Stack is empty, potential JSON substring
                    potential_json = text[start : end + 1]
                    try:
                        objs.append(json.loads(potential_json))
                    except json.JSONDecodeError as exc:
                        errs.append(exc)
                    break
            end += 1
        if prev_num_validations < len(objs) + len(errs):
            start = end + 1  # Start finding the next valid JSON substring.
        else:
            start += 1

    return objs, errs


def extract_jsons(text: str) -> Tuple[List[Dict[str, Any]], List[json.JSONDecodeError]]:
    """
    Extracts the all valid JSON substrings from the given string.

    The function uses a two-pointer approach combined with a stack to efficiently
    identify and extract a valid JSON substring embedded within the input string.

    The list will prefer json strings that are enclosed by ``` or ```json```.

    Parameters:
    - text (str): The input string that potentially contains embedded JSON.

    Returns:
    - List[Dict[str, Any]]: All of extracted valid JSON objects.
    - List[json.JSONDecodeError]: Errors encountered while extracting JSON substrings.

    Raises:
    - ValueError: If no valid JSON substring is found in the input string.
    """
    text = remove_trailing_commas(text.strip())
    text = fix_invalid_escapes(text)
    # Extract all substrings within a ```json``` block or an unlabeled block.
    prefered_blocks = [
        match[1] for match in re.findall(r"```(json)?(.*?)```", text, flags=re.DOTALL)
    ]
    # Extract all substrings outside of a ```json``` block or an unlabeled block
    remaining_blocks = re.split(r"```json.*?```|```.+?```", text, flags=re.DOTALL)
    objs = []
    errs = []
    for block in prefered_blocks + remaining_blocks:
        block_objs, block_errs = _extract_jsons_impl(block.strip())
        objs += block_objs
        errs += block_errs
    return objs, errs
