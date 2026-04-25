# pyre-strict
import json
import tempfile
from typing import Callable


DEFAULT_LENGTH_PER_LINE = 80

DEFAULT_FILE_MESSAGE_TEMPLATE = "(See full content in {filename}. Use file view tools, or bash tools like `head`/`tail`, `cat`, `sed`, `grep` to view and search for specific content)"

DEFAULT_JSON_FILE_MESSAGE_TEMPLATE = "(See full JSON content in {filename}. Prefer bash tools `cat {filename} | jq` to pretty-print and analyze; you can also use file view tools, or bash tools like `head`/`tail`, `sed`, `grep` to search for specific content)"


def _is_json_string(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _default_file_message_builder(filename: str) -> str:
    if filename.endswith(".json"):
        return DEFAULT_JSON_FILE_MESSAGE_TEMPLATE.format(filename=filename)
    return DEFAULT_FILE_MESSAGE_TEMPLATE.format(filename=filename)


def truncate(
    s: str,
    max_lines: int = 100,
    *,
    max_length: int | None = None,
    save_to_file: bool = True,
    prefix: str | None = "confucius_truncate_",
    suffix: str | None = None,
    file_message_builder: Callable[[str], str] | None = None,
) -> str:
    """
    Truncates a string to a maximum number of lines or maximum length, keeping the first part of the string.

    Args:
        s (str): The string to truncate.
        max_lines (int, optional): The maximum number of lines to keep. Defaults to 100.
        max_length (int, optional): The maximum length of the string to keep. Defaults to None.
        save_to_file (bool, optional): Whether to save the truncated string to a file. Defaults to True.
        prefix (str, optional): The prefix to use for the temporary file. Defaults to "confucius_truncate_".
        suffix (str, optional): The suffix to use for the temporary file. Defaults to None and will be auto-detected: `.json` if the content parses as JSON, otherwise `.txt`.
        file_message_builder (Callable[[str], str] | None, optional): A function that takes the filename and returns a message string. If None, a default builder will be used that varies by suffix.

    Returns:
        str: The truncated string with a message indicating that the string was truncated.
    """
    lines = s.splitlines()
    max_length = max_length or max_lines * DEFAULT_LENGTH_PER_LINE

    if max_length is not None and len(s) > max_length:
        # Truncate based on length
        truncated_s = s[:max_length]
        content = (
            truncated_s + f"\n... (truncated {len(s) - max_length} more characters)"
        )
    elif len(lines) > max_lines:
        # Truncate based on line count
        content = (
            "\n".join(lines[:max_lines])
            + f"\n... (truncated {len(lines) - max_lines} more lines)"
        )
    else:
        return s

    # Save to file if needed and truncation occurred
    if save_to_file:
        effective_suffix = (
            suffix
            if suffix is not None
            else (".json" if _is_json_string(s) else ".txt")
        )
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=prefix, suffix=effective_suffix, delete=False
        ) as f:
            f.write(s)
            filename = f.name
        builder = file_message_builder or _default_file_message_builder
        content += f"\n{builder(filename)}"

    return content
