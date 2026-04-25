# pyre-strict


import difflib
import html
import logging
import re
from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel
from rapidfuzz import fuzz

from ...tags import Tag

logger: logging.Logger = logging.getLogger(__name__)

LINE_REGEX: re.Pattern[str] = re.compile(r"\s*(?P<line_num>\d+)\|(?P<content>.*)")
DEFAULT_SIMILARITY_THRESHOLD: float = 0.8


class ChunkWithSimilarity(BaseModel):
    start_line: int
    end_line: int
    similarity: float


def _parse_numbered_lines(content: str) -> list[tuple[int, str]]:
    """
    Parse content with line numbers into (line_number, content) pairs.
    - Ignores whitespace before the line number
    - Preserves all whitespace after the vertical bar
    """
    lines = []
    # Filter out empty or whitespace-only lines, but preserve the content of valid lines exactly
    for line in content.split("\n"):
        # Skip empty lines or lines with only whitespace
        if not line or line.isspace():
            continue
        match = LINE_REGEX.match(line)
        if not match:
            raise ValueError(
                f"Invalid line format: {line}, expected `<line_num>|<content>`"
            )
        line_num = int(match.group("line_num"))
        line_content = match.group("content")
        lines.append((line_num, line_content))
    return lines


def escape_file_content(content: str) -> str:
    """
    Escape HTML special characters in the content of numbered lines.

    Args:
        content (str): The content to be processed, with each line potentially
                       containing a line number followed by a vertical bar and the content.

    Returns:
        str: The content with HTML special characters escaped, preserving line numbers.
    """
    # Use regex to replace </file_edit> with \n</file_edit> only when not preceded by a newline
    content = re.sub(r"(?<!\n)</file_edit>", r"\n</file_edit>", content)

    lines = []
    for line in content.split("\n"):
        match = LINE_REGEX.match(line)
        if match:
            line = html.escape(line, quote=False)
        lines.append(line)
    return "\n".join(lines)


def create_file(path: Path, content: str, require_line_num: bool = True) -> None:
    """Create a new file with the given numbered content."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(
            f"File already exists: `{path}`. Please replace or insert the file content instead."
        )

    if not require_line_num:
        path.write_text(content)
        return

    lines = _parse_numbered_lines(content)

    # Verify line numbers are consecutive starting from 1
    expected_line_num = 1
    for line_num, _ in lines:
        if line_num != expected_line_num:
            raise ValueError(
                f"Line numbers must be consecutive starting from 1. Expected {expected_line_num}, got {line_num}"
            )
        expected_line_num += 1
    # Write only the content part
    path.write_text("\n".join(content for _, content in lines))


def _get_matched_chunk(
    file_path: Path,
    find_lines: list[tuple[int, str]],
    file_lines: list[str],
    similarity_threshold: float,
) -> ChunkWithSimilarity:
    """Get the start and end indices of the matched lines in the file.

    (This function is used for the custom tag approach)

    Args:
        file_path (Path): The path to the file being searched.
        find_lines (list[tuple[int, str]]): A list of tuples where each tuple contains a line number and the corresponding line content to find.
        file_lines (list[str]): A list of strings representing the lines of the file content.
        similarity_threshold (float): The minimum similarity (0 to 1) to consider a match valid.

    Returns:
        ChunkWithSimilarity: The matched chunk. (similarity == 1.0)
    """

    # Verify find lines are consecutive
    for i in range(len(find_lines) - 1):
        if find_lines[i + 1][0] != find_lines[i][0] + 1:
            raise ValueError("Find lines must have consecutive line numbers")
    file_content = "\n".join(file_lines)
    find_text = "\n".join(content for _, content in find_lines)
    matched_chunks = find_matched_chunks_with_similarity(
        find_text=find_text,
        file_content=file_content,
        similarity_threshold=similarity_threshold,
    )

    occurrences = len(matched_chunks)

    if occurrences == 0:
        raise ValueError(
            f"No occurrence found in the file content and no similar part above the similarity threshold {similarity_threshold:.3f}. Please check the file content and try again."
        )

    if occurrences > 1:
        # found the chunk whose line numbers are the closest to the fine_lines line numbers
        matched_chunk = sorted(
            matched_chunks, key=lambda x: abs(x.start_line - find_lines[0][0])
        )[0]
        return matched_chunk

    assert occurrences == 1, "There should be exactly one occurrence at this point"
    matched_chunk = matched_chunks[0]
    if matched_chunk.similarity == 1.0:
        return matched_chunk

    matched_contents = view_file_content(
        content=file_content,
        start_line=matched_chunk.start_line,
        end_line=matched_chunk.end_line,
        max_view_lines=None,
        include_line_numbers=False,
    )
    matched_view = Tag(
        name="view",
        attributes={
            "start_line": str(matched_chunk.start_line),
            "end_line": str(matched_chunk.end_line),
            "file_path": str(file_path),
        },
        contents=view_file_content(
            content=file_content,
            start_line=matched_chunk.start_line,
            end_line=matched_chunk.end_line,
            max_view_lines=None,
        ),
    ).prettify()
    diff_patch = "\n".join(
        difflib.unified_diff(
            find_text.splitlines(),
            matched_contents.splitlines(),
            fromfile="find_text",
            tofile="matched_contents",
        )
    )
    raise ValueError(
        dedent(
            """\
        No exact occurrence found for the search string you provided.
        
        Closest match (similarity: {similarity}):
        {matched_view}
        
        Difference between expected and found text:
        ```
        {diff_patch}
        ```
        
        ACTION REQUIRED: Please update your `<find>` or `<find_after>` tag to match the exact content in the file with `<line_number>|<exact_line_content>` format.
        
        IMPORTANT: YOU MUST CONTINUE USING <file_edit> tag until successful. 
        DO NOT attempt alternative approaches such as:
        - Creating a new file to override the existing one
        - Using command line tools (e.g., `sed`, `awk`, etc.)
        
        Continue refining your `<find>` or `<find_after>` tag until it exactly matches the file content.
        """
        ).format(
            similarity=f"{matched_chunk.similarity:.3f}",
            matched_view=matched_view,
            diff_patch=diff_patch,
        )
    )


def find_matched_chunks_with_similarity(
    find_text: str,
    file_content: str,
    similarity_threshold: float,
) -> list[ChunkWithSimilarity]:
    """
    Search for occurrences of `find_text` in `file_content`.

    - If exact occurrences are found, return a list of tuples with (start_line, end_line, 1.0)
    - If no exact occurrence is found, search for the most similar substring and return a single tuple
      with (start_line, end_line, similarity) if similarity is above threshold
    - If no match above threshold is found, return an empty list

    Args:
      file_content (str): The content of the file.
      find_text (str): The string to find.
      similarity_threshold (float): The minimum similarity (0 to 1) to consider a match valid.

    Returns:
      list[ChunkWithSimilarity]: A list of tuples with (start_line, end_line, similarity) for each match found.
    """
    if len(find_text) == 0:
        raise ValueError("Empty find string provided.")

    # Check for exact matches
    results = []

    # Find all occurrences of the exact text
    start_idx = 0
    while True:
        idx = file_content.find(find_text, start_idx)
        if idx == -1:
            break

        # Calculate line numbers for this match
        start_line = file_content[:idx].count("\n") + 1
        end_line = file_content[: idx + len(find_text)].count("\n") + 1
        results.append(
            ChunkWithSimilarity(
                start_line=start_line, end_line=end_line, similarity=1.0
            )
        )

        # Move past this occurrence
        start_idx = idx + 1

    # If we found exact matches, return them
    if results:
        return results

    # No exact occurrence found; compute the most similar substring.
    n = len(find_text)

    # Handle case where find_text is longer than file_content
    if n > len(file_content):
        return []

    # For fuzzy matching, use partial_ratio_alignment to get position info
    alignment = fuzz.partial_ratio_alignment(
        find_text, file_content, score_cutoff=similarity_threshold * 100
    )

    if alignment is not None:
        best_ratio = alignment.score / 100.0
        match_start = alignment.dest_start
        match_end = alignment.dest_end

        # Calculate line numbers for the match
        start_line = file_content[:match_start].count("\n") + 1
        end_line = file_content[:match_end].count("\n") + 1

        return [
            ChunkWithSimilarity(
                start_line=start_line, end_line=end_line, similarity=best_ratio
            )
        ]

    return []


def _validate_uniqueness(
    file_path: Path,
    find_text: str,
    file_content: str,
    similarity_threshold: float,
) -> None:
    """
    Search for an exact occurrence of `find_text` in `file_content`.

    (This function is used for the tool_use approach)

    - If exactly one exact occurrence is found, return None.
    - If multiple exact occurrences are found, raise an exception stating how many occurrences were found.
    - If no exact occurrence is found, search for the most similar substring (of the same length as find_text)
      within file_content. If its similarity ratio is above the given threshold, raise an exception that
      includes the most similar part and its similarity score. Otherwise, raise an exception stating that
      no occurrence (or sufficiently similar part) was found.

    Args:
      file_path (Path): The path of the file to search in.
      file_content (str): The content of the file.
      find_text (str): The string to find.
      similarity_threshold (float): The minimum similarity (0 to 1) to consider a suggestion valid.

    Raises:
      ValueError: If multiple occurrences are found or if no occurrence (or close enough match) is found.
    """
    matched_chunks = find_matched_chunks_with_similarity(
        find_text, file_content, similarity_threshold
    )
    occurrences = len(matched_chunks)

    if occurrences > 1:
        raise ValueError(
            f"Found {occurrences} occurrences of the text, please include enough context to make it unique"
        )

    if occurrences == 0:
        raise ValueError(
            f"No occurrence found in the file content and no similar part above the similarity threshold {similarity_threshold:.3f}. Please check the file content and try again."
        )

    assert occurrences == 1, "There should be exactly one occurrence at this point"

    matched_chunk = matched_chunks[0]
    if matched_chunk.similarity == 1.0:
        return

    matched_contents = view_file_content(
        content=file_content,
        start_line=matched_chunk.start_line,
        end_line=matched_chunk.end_line,
        max_view_lines=None,
        include_line_numbers=False,
    )
    diff_patch = "\n".join(
        difflib.unified_diff(
            find_text.splitlines(),
            matched_contents.splitlines(),
            fromfile="find_text",
            tofile="matched_contents",
        )
    )
    raise ValueError(
        dedent(
            """\
        No exact occurrence found for the search string you provided.
        
        Closest match (similarity: {similarity}):
        Path: {file_path}
        Line range: {start_line}-{end_line}
        ```
        {matched_contents}
        ```
        
        Difference between expected and found text:
        ```
        {diff_patch}
        ```
        
        ACTION REQUIRED: Please update your `find_text` to match the exact content in the file.
        
        IMPORTANT: YOU MUST CONTINUE USING `str_replace_editor` until successful. 
        DO NOT attempt alternative approaches such as:
        - Creating a new file to override the existing one
        - Using command line tools (e.g., `sed`, `codemod`, etc.)
        
        Continue refining your `find_text` until it exactly matches the file content.
        """
        ).format(
            similarity=f"{matched_chunk.similarity:.3f}",
            file_path=file_path,
            start_line=matched_chunk.start_line,
            end_line=matched_chunk.end_line,
            matched_contents=matched_contents,
            diff_patch=diff_patch,
        )
    )


def _remove_empty_lines(text: str) -> str:
    # IF a line is all space, replace it with an empty line (preserve line count)
    return "\n".join("" if line.strip() == "" else line for line in text.split("\n"))


def _is_full_line_match(content: str, match_text: str, match_index: int) -> bool:
    """Check if the matched text spans complete lines (from start of line to end of line)."""
    if not match_text:
        # An empty string is a full line match if it's at a line boundary
        return (
            match_index == 0
            or content[match_index - 1] == "\n"
            or match_index == len(content)
            or content[match_index] == "\n"
        )

    # Check if match starts at beginning of a line
    starts_at_line_start = match_index == 0 or content[match_index - 1] == "\n"

    # Check if match ends at end of a line
    match_end = match_index + len(match_text)
    ends_at_line_end = match_end == len(content) or content[match_end] == "\n"

    return starts_at_line_start and ends_at_line_end


def _replace_lines_in_original_content(
    file_content: str,
    find_text: str,
    original_file_content: str,
    replace_text: str,
) -> str:
    """
    Replaces a block of text in the original file content.

    This function is designed to be used when the `file_content` has been
    pre-processed (e.g., by removing whitespace-only lines), but the
    replacement needs to happen in the `original_file_content` to preserve
    its structure (like indentation and empty lines).

    Args:
        file_content: The pre-processed content of the file, used for finding
            the location of the text to be replaced.
        find_text: The text to find within `file_content`. It is assumed to
            be a unique, full-line match.
        original_file_content: The original, unprocessed content of the file.
        replace_text: The new text to insert. It is assumed that this text
            is already correctly formatted with the desired indentation.

    Returns:
        The new content with the replacement made in the original context.
    """
    if not find_text:
        return replace_text + original_file_content
    if file_content.count(find_text) != 1:
        return file_content.replace(find_text, replace_text, 1)

    # Find the single match
    match_index = file_content.find(find_text)

    # Check if the match spans complete lines
    if not _is_full_line_match(file_content, find_text, match_index):
        return file_content.replace(find_text, replace_text)

    # Convert character index to line numbers
    lines_before_match = file_content[:match_index].count("\n")
    lines_in_match = find_text.count("\n") + 1

    # Use original content to reconstruct by line numbers
    original_lines = original_file_content.split("\n")
    start_line = lines_before_match
    end_line = start_line + lines_in_match

    # Bounds checking to prevent IndexError
    if end_line > len(original_lines):
        return file_content.replace(find_text, replace_text)

    # Reconstruct using original content to preserve indentation
    new_content = "\n".join(
        original_lines[:start_line]
        + replace_text.split("\n")
        + original_lines[end_line:]
    )
    return new_content


def replace_in_file(
    path: Path,
    find_text: str,
    replace_text: str,
    require_line_num: bool = True,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> None:
    """Replace text in file using line numbers."""

    if not require_line_num:
        original_file_content = path.read_text()
        file_content = _remove_empty_lines(original_file_content)
        find_text = _remove_empty_lines(find_text)

        _validate_uniqueness(
            path, find_text, file_content, similarity_threshold=similarity_threshold
        )

        new_content = _replace_lines_in_original_content(
            file_content=file_content,
            find_text=find_text,
            original_file_content=original_file_content,
            replace_text=replace_text,
        )

        path.write_text(new_content)
        return

    file_content = _remove_empty_lines(path.read_text())
    find_text = _remove_empty_lines(find_text)

    file_lines = file_content.split("\n")

    # Parse find and replace text
    find_lines = _parse_numbered_lines(find_text)
    replace_lines = _parse_numbered_lines(replace_text)

    matched_chunk = _get_matched_chunk(
        path, find_lines, file_lines, similarity_threshold=similarity_threshold
    )
    assert matched_chunk.similarity == 1.0, "Exact match should be found at this point"

    path.write_text(
        "\n".join(
            file_lines[: (matched_chunk.start_line - 1)]
            + [content for _, content in replace_lines]
            + file_lines[matched_chunk.end_line :]
        )
    )


def insert_in_file(
    path: Path,
    find_after: str | None,
    content: str,
    require_line_num: bool = True,
    insert_line: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> None:
    """
    Insert content after specified line(s).
    - <find_after> can be multiple consecutive lines that must match exactly
    - Content will be inserted after the last matched line
    """
    file_content = path.read_text()
    file_lines = file_content.split("\n")

    if find_after is not None:
        find_after = _remove_empty_lines(find_after)

    if not require_line_num:
        if insert_line is not None:
            path.write_text(
                "\n".join(
                    file_lines[:insert_line]
                    + content.split("\n")
                    + file_lines[insert_line:]
                )
            )
        else:
            if find_after is None:
                raise ValueError("Must specify find_after when not using `insert_line`")

            _validate_uniqueness(
                path,
                find_after,
                file_content,
                similarity_threshold=similarity_threshold,
            )
            new_content = file_content.replace(find_after, f"{find_after}\n{content}")
            path.write_text(new_content)
        return

    # Parse find_after and content
    insert_lines = _parse_numbered_lines(content)

    if insert_line is None:
        if find_after is None:
            raise ValueError("Must specify find_after when not using `insert_line`")

        find_lines = _parse_numbered_lines(find_after)
        matched_chunk = _get_matched_chunk(
            path, find_lines, file_lines, similarity_threshold=similarity_threshold
        )
        assert (
            matched_chunk.similarity == 1.0
        ), "Exact match should be found at this point"

        insert_line = matched_chunk.end_line

    path.write_text(
        "\n".join(
            file_lines[:insert_line]
            + [content for _, content in insert_lines]
            + file_lines[insert_line:]
        )
    )


def delete_file(path: Path) -> None:
    """Delete a file."""
    if path.exists():
        path.unlink()
    else:
        raise FileNotFoundError(
            f"File not found: {path}. Please use the absolute path to the file. Your current working directory is {Path.cwd().resolve()}."
        )


def view_file(
    path: Path, start_line: int | None, end_line: int | None, max_view_lines: int | None
) -> str:
    """View file contents with line numbers and return as a string.

    Args:
        path (Path): Path to the file to view.
        start_line (int, optional): Line number to start viewing from. Defaults to None (start of file).
        end_line (int, optional): Line number to end viewing at. Defaults to None (end of file).
        max_view_lines (int, optional): Maximum number of lines to view. Defaults to None (no limit).

    Returns:
        str: Contents of the file with line numbers, aligned for readability.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Please use the absolute path to the file. Your current working directory is {Path.cwd().resolve()}."
        )

    return view_file_content(path.read_text(), start_line, end_line, max_view_lines)


def view_file_content(
    content: str,
    start_line: int | None,
    end_line: int | None,
    max_view_lines: int | None,
    include_line_numbers: bool = True,
) -> str:
    """View file contents with line numbers and return as a string.

    Args:
        content (str): Content of the file to view.
        start_line, end_line, max_view_lines (int, optional): See `view_file`.

    Returns:
        str: Contents of the file with line numbers, aligned for readability.
    """
    lines = content.split("\n")

    if start_line is None:
        start_line = 1
    if end_line is None or end_line == -1:
        end_line = len(lines)

    if start_line < 1 or end_line < 1:
        raise ValueError("Line numbers must be positive")

    if start_line > end_line:
        raise ValueError("Start line must be less than or equal to end line")

    if max_view_lines is not None and end_line - start_line + 1 > max_view_lines:
        raise ValueError(
            f"Max view lines exceeded: {end_line - start_line + 1} > {max_view_lines}, please specify a smaller range"
        )

    # Calculate width needed for line numbers
    width = len(str(end_line))

    result = []
    # Prepend # lines before start_line
    if start_line > 1:
        result.append(f"({start_line-1} line(s) above)")

    # Add line numbers and content to result
    for i, line in enumerate(lines, start=1):
        if i >= start_line and i <= end_line:
            if include_line_numbers:
                result.append(f"{i:{width}}|{line}")
            else:
                result.append(line)

    # Append # lines after end_line
    if end_line < len(lines):
        result.append(f"({len(lines)-end_line} line(s) below)")

    return dedent("\n".join(result))


def view_directory(path: Path, depth: int, show_hidden: bool) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if depth < 1:
        raise ValueError("Depth must be positive")

    base_pattern = "[!.]*" if not show_hidden else "*"
    patterns = [base_pattern]
    for d in range(1, depth):
        patterns.append(f"{base_pattern}{'/*' * d}")

    result = []
    for pattern in patterns:
        for item in sorted(path.glob(pattern)):
            if item.is_file():
                result.append(str(item))
            else:
                result.append(f"{item}/")

    return "\n".join(sorted(result))
