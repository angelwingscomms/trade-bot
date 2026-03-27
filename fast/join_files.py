#!/usr/bin/env python3
"""
Combines all files in the fast/ directory (except FLAWS.md) into a single pipeline.md file.
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
    
    # Files to exclude
    exclude_files = {'FLAWS.md', 'combine_to_pipeline.py', 'pipeline.md', 'join_files.py'}
    
    # Extensions to include (text files only)
    text_extensions = {'.mq5', '.py', '.txt', '.csv', '.json', '.md', '.yaml', '.yml', 
                       '.xml', '.html', '.css', '.js', '.ts', '.sql', '.sh', '.bat', '.ps1'}
    
    # Get all files in the directory (not recursive)
    files = []
    for item in script_dir.iterdir():
        if item.is_file() and item.name not in exclude_files:
            # Only include files with known text extensions
            if item.suffix.lower() in text_extensions:
                files.append(item)
    
    # Sort files for consistent output
    files.sort(key=lambda x: x.name)
    
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
