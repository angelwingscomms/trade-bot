from __future__ import annotations

from .shared import *  # noqa: F401,F403


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Combine files into pipeline.md")
    parser.add_argument(
        "-i",
        "--include-flaws",
        action="store_true",
        help="Include FLAWS.md in the output",
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default=None,
        help="Specify subfolder to read files from (e.g., bitcoin)",
    )
    args = parser.parse_args()

    # Get the directory this script is in
    script_dir = Path(__file__).resolve().parents[3]

    # Determine the source directory
    if args.root:
        source_dir = script_dir / args.root
        if not source_dir.exists():
            print(f"Error: Folder '{args.root}' does not exist")
            return
        if not source_dir.is_dir():
            print(f"Error: '{args.root}' is not a directory")
            return
    else:
        source_dir = script_dir

    # Include these specific files (relative to source_dir / project root)
    include_files = [
        "mt5/scripts/data.mq5",
        "scripts/nn.py",
        "live/live.mq5",
    ]

    # Optionally include docs/notes/FLAWS.md or root FLAWS.md
    if args.include_flaws:
        include_files.append("docs/notes/FLAWS.md")

    # Get the specified files
    files = [
        source_dir / name for name in include_files if (source_dir / name).exists()
    ]

    # Build the pipeline.md content
    output_lines = []

    for file_path in files:
        print(f"Processing: {file_path.name}")

        # Read file content (UTF-8 only for text files)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Get language tag
        lang_tag = get_language_tag(file_path.name)

        # Add to output
        output_lines.append(file_path.name)
        output_lines.append(f"```{lang_tag}")
        output_lines.append(content.rstrip())
        output_lines.append("```")
        output_lines.append("")  # Empty line between files

    # Write pipeline.md to the source directory
    output_path = source_dir / "pipeline.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\nCreated {output_path}")
    print(f"Combined {len(files)} file(s)")
