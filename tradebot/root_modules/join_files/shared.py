#!/usr/bin/env python3
"""
Combines mt5/scripts/data.mq5, scripts/nn.py, and live/live.mq5 into a
single pipeline.md file. Each file is wrapped in a code block with
appropriate language tagging.
Use -i flag to also include FLAWS.md.
Use -r flag to specify a subfolder (e.g., -r bitcoin).
"""

import argparse
from pathlib import Path

# Mapping of file extensions to markdown language identifiers
LANG_MAP = {
    ".mq5": "cpp",
    ".py": "python",
    ".txt": "",
    ".csv": "",
    ".json": "json",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
    ".sql": "sql",
    ".sh": "bash",
    ".bat": "batch",
    ".ps1": "powershell",
}
