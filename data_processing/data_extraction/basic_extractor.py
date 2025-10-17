import argparse
import yaml
import numpy as np
import h5py
from pathlib import Path
from rosbags.highlevel import AnyReader

from data_processing.data_extraction.hdf5_utils import ensure_group, create_expandable_dataset
from data_processing.data_extraction.label_extractor import insert_labels
from data_processing.data_extraction.environment_extractor import insert_environment


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_to_hdf5(cfg):
    bag_path = Path(cfg["bag"]["path"])
    output_path = Path(cfg["output"]["path"])

    raw_topic = cfg["topics"]["raw_detections"]
    fused_topic = cfg["topics"]["fused_detections"]
    image_topic = cfg["topics"]["rgb_image"]
    filter_ids = cfg["filters"]["tracking_ids"]

    with AnyReader([bag_path]) as reader, h5py.File(output_path, "w") as hdf:
        # ------------------ Metadata ------------------
        meta = ensure_group(hdf, "metadata")
        meta.create_dataset("joint_names", data=np.array([f"joint_{i}" for i in range(32)], dtype=h5py.string_dtype()))
        meta.attrs["bag_start_time"] = reader.start_time / 1e9
        meta.attrs["label_anchor"] = "bag_start_time"

        poses_raw = ensure_group(hdf, "poses_raw")
        poses_fused = ensure_group(hdf, "poses_fused")
        images = ensure_group(hdf, "images")
        image_topic_group = ensure_group(images, image_topic.replace("/", "_"))

        image_timestamps = []
        extracted_ids = set()

        # Filter connections
        connections = {c.topic: c for c in reader.connections if c.topic in [raw_topic, fused_topic, image_topic]}

        # ------------------ Process Messages ------------------
        for conn, timestamp, rawdata in reader.messages(connections=connections.values()):
            ts = timestamp / 1e9

            # Handle image timestamps
            if conn.topic == image_topic:
                image_timestamps.append(ts)
                continue

            msg = reader.deserialize(rawdata, conn.msgtype)
            num_joints = 32
            if len(msg.points) % num_joints != 0:
                continue

            num_detections = len(msg.points) // num_joints
            channel_map = {c.name.lower(): c.values for c in msg.channels}

            tracking_ids = channel_map.get("tracking_id", [])
            track_states = channel_map.get("track_state", None)
            source_ids = channel_map.get("source_id", [])

            for det_idx in range(num_detections):
                start_idx = det_idx * num_joints
                end_idx = start_idx + num_joints
                tid = int(tracking_ids[start_idx])
                sid = int(source_ids[start_idx]) if len(source_ids) else -1

                if tid not in filter_ids:
                    continue

                joints = np.array([[j.x, j.y, j.z] for j in msg.points[start_idx:end_idx]], dtype="f4")
                track_state = float(track_states[start_idx]) if track_states is not None else np.nan

                # RAW structure
                if conn.topic == raw_topic:
                    raw_tid_group = ensure_group(poses_raw, str(tid))
                    src_group = ensure_group(raw_tid_group, f"source_{sid}")
                    tid_group = ensure_group(src_group, str(tid))

                # FUSED structure
                elif conn.topic == fused_topic:
                    tid_group = ensure_group(poses_fused, str(tid))
                else:
                    continue

                # Create datasets if needed
                create_expandable_dataset(tid_group, "timestamps", (0,), "f8")
                create_expandable_dataset(tid_group, "joints", (0, num_joints, 3), "f4")
                create_expandable_dataset(tid_group, "detection_indices", (0,), "i4")
                create_expandable_dataset(tid_group, "track_states", (0,), "f4")

                # Append new row
                tid_group["timestamps"].resize((tid_group["timestamps"].shape[0] + 1,))
                tid_group["timestamps"][-1] = ts

                tid_group["joints"].resize((tid_group["joints"].shape[0] + 1, num_joints, 3))
                tid_group["joints"][-1] = joints

                tid_group["detection_indices"].resize((tid_group["detection_indices"].shape[0] + 1,))
                tid_group["detection_indices"][-1] = det_idx

                tid_group["track_states"].resize((tid_group["track_states"].shape[0] + 1,))
                tid_group["track_states"][-1] = track_state

                extracted_ids.add(tid)

        # ------------------ Finalize Metadata ------------------
        meta.create_dataset("extracted_ids", data=np.array(sorted(extracted_ids), dtype="i4"))
        if len(image_timestamps) > 0:
            image_topic_group.create_dataset("timestamps", data=np.array(image_timestamps, dtype="f8"))
            meta.attrs["video_start_time"] = image_timestamps[0]

        # ------------------ Insert TMV labels if available ------------------
        labels_cfg = cfg.get("labels", {})
        tmv_file = labels_cfg.get("tmv_file", None)
        mapping = {str(k): v for k, v in labels_cfg.get("mapping", {}).items()}

        if tmv_file and Path(tmv_file).exists() and mapping:
            print(f"[INFO] Merging TMV labels from {tmv_file}")
            insert_labels(hdf, tmv_file, mapping)
            meta.attrs["tmv_file"] = str(tmv_file)
            meta.attrs["labels_are_relative"] = False  # since we use absolute timestamps
        else:
            print("[INFO] No TMV label file or mapping found — skipping label insertion.")

        # ------------------ Insert environment if available ------------------
        env_cfg = cfg.get("environment", {})
        env_json = env_cfg.get("json_file", None)
        if env_json and Path(env_json).exists():
            print(f"[INFO] Inserting environment from {env_json}")
            insert_environment(hdf, env_json)
        else:
            print("[INFO] No environment JSON found — skipping environment insertion.")

    print(f"Extraction complete: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract raw and fused skeletons from ROS bag into HDF5.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    extract_to_hdf5(cfg)

# Plan for basic_extractor.py - Phase 1

# 1. Load config YAML.
# 2. Open ROS bag with `AnyReader`.
# 3. Create HDF5 groups: `metadata`, `poses_raw`, `poses_fused`, `images`.
# 4. For each message:
#    • If image → append timestamp.
#    • If raw or fused detection → extract tracking ID, source ID, joints, timestamps, track state, write using `hdf5_utils`.
# 5. After looping, store `extracted_ids`, `joint_names`, and basic metadata attributes (bag_start_time, video_start_time, label_anchor = bag_start_time).

# No labels, environment, or sensors yet. The fusion logic uses simple per-joint averaging of same-timestamp data from multiple sensors.