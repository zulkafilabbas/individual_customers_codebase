#!/usr/bin/env python3
import os
import json
from common_utils.loader import JsonLoader

# Initialize loader (reads JSONs from common_utils/)
loader = JsonLoader("common_utils")

# ----------------------------------------------------
# Settings
# ----------------------------------------------------

# Get settings from loader
settings = loader.get_data_extraction_settings()
BASE_DIR = loader.get_data_extraction_settings()["BASE_DIR"]
CONFIG_OUT = loader.get_data_extraction_settings()["CONFIG_OUT"]
DATASET_OUT = loader.get_data_extraction_settings()["DATASET_OUT"]

TOPICS_DEFAULT = {
    "raw_detections": "/tracked_detections",
    "fused_detections": "/tracked_detections_fused",
    "rgb_image": "/windows_02/image/compressed",
}

TOPICS_FIX_SORTED = {
    "raw_detections": "/tracked_detections_corrected",
    "fused_detections": "/tracked_detections_fused_corrected",
    "rgb_image": TOPICS_DEFAULT["rgb_image"],
}

def to_yaml_path(path: str) -> str:
    """Normalize to absolute path and use forward slashes."""
    return os.path.abspath(path).replace("\\", "/")

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def write_config(entry, out_dir, env_json_path, extrinsics_year="2025"):
    bag_file = entry["Bag File"]
    tmv_file = entry["Tmv File"]
    track_id = entry["Track ID"]

    is_fix_sorted = "fix_sorted" in bag_file

    bag_path = to_yaml_path(os.path.join(BASE_DIR, bag_file))
    tmv_path = to_yaml_path(os.path.join(BASE_DIR, tmv_file))
    output_path = to_yaml_path(os.path.join(DATASET_OUT, os.path.basename(bag_file).replace(".bag", ".hdf5")))

    topics_dict = TOPICS_FIX_SORTED if is_fix_sorted else TOPICS_DEFAULT

    topics_yaml = "\n".join([f'  {k}: "{v}"' for k, v in topics_dict.items()])

    extrinsics = loader.get_extrinsics(extrinsics_year)
    extrinsics_json_path = to_yaml_path(os.path.join("common_utils", "hatshop_extrinsic_calibration.json"))

    content = f"""bag:
  path: "{bag_path}"

output:
  path: "{output_path}"

topics:
{topics_yaml}
labels:
  tmv_file: "{tmv_path}"
  mapping:
    {track_id}: {track_id}

filters:
  tracking_ids: [{track_id}]

environment:
  json_file: "{env_json_path}"

extrinsics:
  json_file: "{extrinsics_json_path}"
  year: {extrinsics_year}
"""

    cfg_name = bag_file.replace(".bag", "_config.yaml")
    cfg_path = os.path.join(out_dir, cfg_name)

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)

    return cfg_path

# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    os.makedirs(CONFIG_OUT, exist_ok=True)
    os.makedirs(DATASET_OUT, exist_ok=True)

    # load from the shared JSON index
    index = loader.get_customers()
    entries = index.get("entries", [])

    ok_count = 0
    skip_count = 0
    env_json_path = to_yaml_path(os.path.join("common_utils", "hatshop_environment_objects.json"))

    for entry in entries:
        bag_file = entry.get("Bag File", "").strip()
        tmv_file = entry.get("Tmv File", "").strip()
        track_id = entry.get("Track ID", -1)

        if not bag_file or not tmv_file or track_id == -1:
            skip_count += 1
            continue

        bag_path = os.path.join(BASE_DIR, bag_file)
        tmv_path = os.path.join(BASE_DIR, tmv_file)
        if not (os.path.exists(bag_path) and os.path.exists(tmv_path)):
            skip_count += 1
            continue

        cfg_path = write_config(entry, CONFIG_OUT, env_json_path)
        print(f"[OK] Wrote config: {cfg_path}")
        ok_count += 1

    print(f"\nSummary: {ok_count} configs written, {skip_count} skipped.")

if __name__ == "__main__":
    main()
