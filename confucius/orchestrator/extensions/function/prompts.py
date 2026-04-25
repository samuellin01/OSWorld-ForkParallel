# pyre-strict

from textwrap import dedent

FUNCTION_CALL_BASIC_PROMPT: str = dedent(
    """\
    You are capable of making structured function calls.

    Some general guidelines for function calls:
    - Wrap the function call in <{function_call_tag_name}> tags
    - Include a "name" attribute specifying the function name
    - Provide function arguments as a JSON payload within the tag
    - Assign an identifier to the `identifier` attribute of the opening <{function_call_tag_name}> tag. For updates, reuse the prior identifier. Make sure you use the same identifier for all updates to the same JSON blob. For new JSON blob, the identifier should be descriptive and relevant to the content, using kebab-case (e.g., "example-function-call"). This identifier will be used consistently throughout the JSON blob's lifecycle, even when updating or iterating on the JSON.

    For example:
    <{function_call_tag_name} name="function_name" identifier="example-function-call">
    {{
        "arg1": value1,
        "arg2": value2,
        ...
    }}
    </{function_call_tag_name}>

    Here are all the available functions:
    {functions}
    """
)
