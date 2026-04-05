"""
Execute data.mq5 via MT5 terminal and move output to ./data/SYMBOL/ticks.csv
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

# MT5 Configuration
TERMINAL_PATH = Path(r"C:\Program Files\MetaTrader 5\terminal64.exe")
MT5_INSTANCE_ROOT = Path(r"C:\Users\Admin\AppData\Roaming\MetaTrader 5")  # Adjust if needed
EXPERTS_DIR = MT5_INSTANCE_ROOT / "MQL5" / "Experts" / "trade-bot"
FILES_DIR = MT5_INSTANCE_ROOT / "Files"

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data"


def get_symbol() -> str:
    """Extract SYMBOL from shared_config.mqh"""
    config_path = SCRIPT_DIR / "shared_config.mqh"
    with open(config_path, "r") as f:
        for line in f:
            if '#define SYMBOL' in line:
                # Extract symbol from: #define SYMBOL "XAUUSD"
                symbol = line.split('"')[1]
                return symbol
    raise ValueError("Could not find SYMBOL in shared_config.mqh")


def find_csv_file(symbol: str, max_wait: float = 30.0) -> Optional[Path]:
    """
    Wait for the script output CSV file to appear in MT5 Files folder.
    
    Args:
        symbol: The trading symbol
        max_wait: Maximum seconds to wait for file
        
    Returns:
        Path to the CSV file, or None if not found
    """
    start_time = time.time()
    
    # data.mq5 defaults to "market_ticks.csv" 
    csv_name = "market_ticks.csv"
    csv_path = FILES_DIR / csv_name
    
    print(f"[INFO] Waiting for {csv_path}...")
    
    while time.time() - start_time < max_wait:
        if csv_path.exists():
            # Give it a moment to finish writing
            time.sleep(0.5)
            if csv_path.stat().st_size > 100:  # Ensure it has content
                print(f"[INFO] Found: {csv_path}")
                return csv_path
        time.sleep(0.5)
    
    return None


def run_script(symbol: str) -> bool:
    """
    Run data.mq5 script via MT5 terminal.
    
    Args:
        symbol: The trading symbol to export
        
    Returns:
        True if script execution was triggered
    """
    if not TERMINAL_PATH.exists():
        print(f"[ERROR] MT5 terminal not found: {TERMINAL_PATH}")
        return False
    
    print(f"[INFO] Launching MT5 with data.mq5 script...")
    print(f"[INFO] Symbol: {symbol}")
    
    try:
        # Run terminal with script
        # /script parameter runs the script in the Experts folder
        proc = subprocess.Popen(
            [str(TERMINAL_PATH), "/script=trade-bot::data"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"[INFO] Terminal started (PID: {proc.pid})")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to launch terminal: {e}")
        return False


def move_to_data_dir(symbol: str, csv_path: Path) -> Path:
    """
    Move CSV file to ./data/SYMBOL/ticks.csv
    
    Args:
        symbol: Trading symbol
        csv_path: Source CSV file path
        
    Returns:
        Path to the destination file
    """
    symbol_dir = OUTPUT_DIR / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = symbol_dir / "ticks.csv"
    
    print(f"[INFO] Moving {csv_path.name} to {dest_path}...")
    shutil.move(str(csv_path), str(dest_path))
    
    file_size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Successfully moved file ({file_size_mb:.2f} MB)")
    print(f"[INFO] Output: {dest_path}")
    
    return dest_path


def main():
    """Main execution flow"""
    print("=" * 60)
    print("MT5 Data Export Tool")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Get symbol from config
        print("[STEP 1] Reading configuration...")
        symbol = get_symbol()
        print(f"[STEP 1] SUCCESS - Symbol: {symbol}")
        print()
        
        # Step 2: Run MT5 script
        print("[STEP 2] Launching MT5 terminal with data.mq5...")
        if not run_script(symbol):
            print("[STEP 2] FAILED - Could not launch terminal")
            return False
        print("[STEP 2] SUCCESS")
        print()
        
        # Step 3: Wait for output file
        print("[STEP 3] Waiting for script output (up to 30 seconds)...")
        csv_path = find_csv_file(symbol, max_wait=30.0)
        
        if csv_path is None:
            print("[STEP 3] FAILED - Timeout waiting for CSV file")
            print("[INFO] Possible issues:")
            print(f"  - MT5 Files folder: {FILES_DIR}")
            print("  - Check MT5 Experts log for errors")
            print("  - Symbol may not have tick history downloaded")
            return False
        print("[STEP 3] SUCCESS")
        print()
        
        # Step 4: Move to data directory
        print("[STEP 4] Moving file to data directory...")
        dest_path = move_to_data_dir(symbol, csv_path)
        print("[STEP 4] SUCCESS")
        print()
        
        print("=" * 60)
        print("Export completed successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
