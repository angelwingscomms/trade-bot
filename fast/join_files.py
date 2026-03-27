#!/usr/bin/env python3
"""
Combines data.mq5, nn.py, and live.mq5 into a single pipeline.md file.
Each file is wrapped in a code block with appropriate language tagging.
"""

from pathlib import Path

# Mapping of file extensions to markdown language identifiers
LANG_MAP = {
    '.mq5': 'cpp',
    '.py': 'python',
    '.txt': '',
    '.csv': '',
    '.json': 'json',
    '.md': 'markdown',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.xml': 'xml',
    '.html': 'html',
    '.css': 'css',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.sql': 'sql',
    '.sh': 'bash',
    '.bat': 'batch',
    '.ps1': 'powershell',
}

def get_language_tag(filename: str) -> str:
    """Get the markdown language tag based on file extension."""
    ext = Path(filename).suffix.lower()
    return LANG_MAP.get(ext, '')

def main():
    # Get the directory this script is in
    script_dir = Path(__file__).parent.resolve()
    
    # Only include these specific files
    include_files = ['data.mq5', 'nn.py', 'live.mq5']
    
    # Get the specified files
    files = [script_dir / name for name in include_files if (script_dir / name).exists()]
    
    # Build the pipeline.md content
    output_lines = []
    
    for file_path in files:
        print(f"Processing: {file_path.name}")
        
        # Read file content (UTF-8 only for text files)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get language tag
        lang_tag = get_language_tag(file_path.name)
        
        # Add to output
        output_lines.append(f"```{lang_tag}")
        output_lines.append(content.rstrip())
        output_lines.append("```")
        output_lines.append("")  # Empty line between files
    
    # Write pipeline.md
    output_path = script_dir / 'pipeline.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✅ Created {output_path}")
    print(f"   Combined {len(files)} file(s)")

if __name__ == '__main__':
    main()
