"""
Post-processing script to move achilles_ticks.csv from MQL5 sandbox
to the project directory.

MQL5 restricts file writing to MQL5\Files directory, so this script
moves the exported file to MQL5\Experts\nn\fast after export.
"""

import shutil
import os
from pathlib import Path

# Define paths
# Terminal data path
TERMINAL_PATH = Path(r"C:\Users\edhog\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075")

# Source: MQL5 sandbox location
SOURCE_FILE = TERMINAL_PATH / "MQL5" / "Files" / "fast" / "achilles_ticks.csv"

# Destination: project directory
DEST_FILE = TERMINAL_PATH / "MQL5" / "Experts" / "nn" / "fast" / "achilles_ticks.csv"

def move_ticks_file():
    """Move the ticks file from sandbox to project directory."""
    
    print("=" * 60)
    print("ACHILLES TICKS FILE MOVER")
    print("=" * 60)
    
    # Check if source file exists
    if not SOURCE_FILE.exists():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        return False
    
    # Get file size
    file_size = SOURCE_FILE.stat().st_size
    print(f"Source file: {SOURCE_FILE}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    # Ensure destination directory exists
    DEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Move the file (copy then delete source)
    print(f"\nMoving to: {DEST_FILE}")
    
    try:
        shutil.copy2(SOURCE_FILE, DEST_FILE)
        print("File copied successfully")
        
        # Verify the copy
        if DEST_FILE.exists() and DEST_FILE.stat().st_size == file_size:
            # Delete the source file
            SOURCE_FILE.unlink()
            print("Source file deleted")
            print("=" * 60)
            print("SUCCESS: File moved to project directory")
            print("=" * 60)
            return True
        else:
            print("ERROR: Destination file size mismatch")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to move file: {e}")
        return False

if __name__ == "__main__":
    move_ticks_file()
