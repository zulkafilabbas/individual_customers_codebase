#!/usr/bin/env python3
"""
batch_extract.py
--------------------------------
Runs basic_extractor.py sequentially for all YAML config files
under the configs/ directory.
--------------------------------
"""

import subprocess
from pathlib import Path

CONFIG_DIR = Path("configs")

def run_extractor(config_path):
    print(f"\n[INFO] Running extractor for: {config_path}")
    try:
        subprocess.run(
            ["python", "-m", "data_processing.data_extraction.basic_extractor", "--config", str(config_path)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Extraction failed for {config_path}: {e}")

def main():
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    if not configs:
        print("[ERROR] No YAML configs found in configs/ directory.")
        return

    print(f"[INFO] Found {len(configs)} configs.")
    for cfg in configs:
        run_extractor(cfg)

if __name__ == "__main__":
    main()
