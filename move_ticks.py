from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILES_DIR = PROJECT_ROOT.parent / "Files"
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Move tick files from MT5 Files folder to ./data")
    parser.add_argument("-i", "--input", required=True, help="Filename to move")
    args = parser.parse_args()

    source_path = FILES_DIR / args.input
    dest_path = DATA_DIR / args.input

    print(f"[INFO] Looking for {args.input} in {FILES_DIR}...")
    if not source_path.exists():
        print(f"[ERROR] Source file not found: {source_path}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Moving to {dest_path}...")

    try:
        shutil.copy2(source_path, dest_path)
        source_path.unlink()
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"[INFO] Successfully moved {args.input} to ./data/")
        print(f"[INFO] File size: {size_mb:.2f} MB")
    except Exception as exc:
        print(f"[ERROR] Failed to move file: {exc}")


if __name__ == "__main__":
    main()