# pyre-strict
from __future__ import annotations

from typing import Dict

# Return a mapping of allowed command -> description string
# Keep conservative; validators in CommandLineExtension will do further checks


def get_allowed_commands() -> Dict[str, str]:
    return {
        # Filesystem basics
        "pwd": "Print current working directory",
        "ls": "List directory contents",
        "cat": "Print file contents",
        "head": "Show first lines of a file",
        "tail": "Show last lines of a file",
        "wc": "Count lines/words/bytes",
        "stat": "Display file or file system status",
        "du": "Estimate file space usage",
        "df": "Report file system disk space usage",
        "chmod": "Change file access permissions",
        "chown": "Change file ownership",
        "cp": "Copy files and directories",
        "mv": "Move (rename) files",
        "rm": "Remove (delete) files or directories",
        "mkdir": "Create directories",
        "rmdir": "Remove empty directories",
        "touch": "Update access and modification times of files",
        "find": "Search for files in a directory hierarchy",
        # Text processing
        "grep": "Search for patterns in files",
        "sed": "Stream editor for filtering and transforming text",
        "awk": "Text processing and data extraction",
        "cut": "Remove sections from each line of files",
        "sort": "Sort lines of text files",
        "uniq": "Report or omit repeated lines",
        "tr": "Translate or delete characters",
        "xargs": "Build and execute command lines from standard input",
        # Archiving / compression
        "tar": "Archive files",
        "gzip": "Compress or decompress named files",
        # Networking (safe reads only)
        "curl": "Transfer data from or to a server",
        "wget": "Non-interactive network downloader",
        # Git (single entry for all common subcommands)
        "git": "Git version control (status, diff, add, commit, branch, checkout, switch, log, show, grep, rev-parse, etc.)",
        # Python runner
        "python3": "Run Python scripts",
    }
