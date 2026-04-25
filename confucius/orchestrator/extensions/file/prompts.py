# pyre-strict

from textwrap import dedent

FILE_EDIT_BASIC_PROMPT: str = dedent(
    """\
    You are capable of editing and viewing files. The edits are performed in sequence and state is persistent across commands. Here's how to use each operation:

    Important notes about line format and whitespace:
    - Format is: "[optional_spaces]<line_number>|<exact_line_content>"
    - Whitespace before the line number is ignored
    - The vertical bar (|) represents the exact boundary of the file content
    - Any whitespace after | is considered part of the line content and will be preserved
    - Examples of valid line formats:
      * "1|No leading space", parsed line number is 1, content is "No leading space"
      * "  2|  Two spaces at start", parsed line number is 2, content is "  Two spaces at start"
      * "    27|Text with spaces  ", parsed line number is 27, content is "Text with spaces  "
      * "1234|Content after any line number", parsed line number is 1234, content is "Content after any line number"


    1. CREATE a new file:
    - Line numbers must start from 1 and be consecutive
    - Each line format is "<line_number>|<exact_line_content>"
    - Whitespace after | is preserved exactly as written
    - Format:
    
    <file_edit type="create" file_path="/absolute/path/to/new/file.ext">
    1|Content of the new file
    2|  Including leading whitespace
    3|And trailing spaces  
    </file_edit>
    

    2. REPLACE content in existing file:
    - Replacements are line-based only - partial line replacements are not supported
    - Line numbers in <find> must match exactly the lines in the file to be replaced
    - Lines in <find> must be consecutive
    - Whitespace after | must match the file exactly
    - <replace> can contain any number of lines
    - Each line format is "<line_number>|<exact_line_content>", both for <find> and <replace>
    - Format:
    
    <file_edit type="replace" file_path="/absolute/path/to/file.ext">
    <find>
    5|    Line to find with exact spaces
    6|No leading space line
    7|  Another indented line  
    </find>
    <replace>
    5|    Preserving same indentation
    6|New content can have different spacing
    </replace>
    </file_edit>
    

    3. INSERT content after a specific line:
    - Insertions are line-based only - inserting within a line is not supported
    - <find_after> specifies where to insert the content
    - Recommended: Use single line in <find_after> for simplicity and clarity
    - The line(s) in <find_after> must match exactly (including whitespace)
    - Content will be inserted after the last matched line
    - Each line format is "<line_number>|<exact_line_content>", both for <find_after> and <content>
    - Format:

    
    <file_edit type="insert" file_path="/absolute/path/to/file.ext">
    <find_after>
    5|    Exact line with spaces to insert after
    </find_after>
    <content>
    6|New content preserves spaces too
    7|  Including indentation  
    </content>
    </file_edit>
    

    4. DELETE a file:
    - Format:
    
    <file_edit type="delete" file_path="/absolute/path/to/file.ext">
    </file_edit>
    

    5. VIEW file content:
    - Optionally specify line range if the file is too long
    - If you specify line range:
      - Always specify sufficient line range to see complete context
      - For code files, include surrounding context (e.g., entire function or class)
      - If unsure, include extra lines before and after the target area
    - Format:
    
    <file_edit type="view" file_path="/absolute/path/to/file.ext" start_line="1" end_line="10">
    </file_edit>

    In the file view result, you will see the file content with line numbers. The format will be like:
    <view [attributes ...] >
     1|<file content line 1>
     2|<file content line 2>
    ...
    10|<file content line 10>
    </view>
    

    6. VIEW directory:
    - Optionally specify depth and whether to show hidden files
    - Format:
    
    <file_edit type="view" directory="/absolute/path/to/dir" depth="2" show_hidden>
    </file_edit>
    
    or

    <file_edit type="view" directory="/absolute/path/to/dir" depth="2">
    </file_edit>
    

    7. UNDO last change:
    - Reverts the most recent edit made to the specified file
    - Format:
    
    <file_edit type="undo" file_path="/absolute/path/to/file.ext">
    </file_edit>
    

    Important Notes:
    - Immediately before <file_edit> operation call, think for one sentence in <thinking> tags about the thought process regarding the operation.
    - State is persistent across commands and conversations
    - Use absolute paths
    - CREATE will fail if file already exists
    - Long outputs will be truncated
    - For REPLACE:
      * <find> must match lines EXACTLY (including whitespace)
      * Match must be unique in file
      * Include enough context in <find> to ensure uniqueness
    - UNDO reverts the most recent change to the specified file
    - Use proper XML escaping for special characters
    """
)


FILE_EDIT_TOOL_USE_USER_ERR_MESSAGE: str = dedent(
    """\
    Please think and plan carefully about how to fix the error.
    IMPORTANT: YOU MUST CONTINUE USING `str_replace_editor` with the same command until successful.
    """
)


FILE_EDIT_TOOL_USER_ACCESS_DENIED_MESSAGE: str = dedent(
    """\
    You have been explicitly denied access to the file or directory.
    Please think and plan carefully about how to solve the original problem.
    """
)
