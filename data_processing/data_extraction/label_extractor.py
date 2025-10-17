import yaml
import numpy as np
import h5py

def insert_labels(hdf: h5py.File, tmv_file: str, mapping: dict):
    """
    Insert TMV labels into the given HDF5 file using a label-to-tracking_id mapping.
    This preserves your exact legacy behavior for timing and alignment.

    Parameters
    ----------
    hdf : h5py.File
        Open HDF5 handle containing /metadata group with bag_start_time attr.
    tmv_file : str
        Path to the .tmv YAML file.
    mapping : dict
        Mapping from TMV label IDs (as strings) to HDF5 tracking IDs (ints).
    """
    print(f"[TMV] Loading: {tmv_file}")
    with open(tmv_file, "r", encoding="utf-8", errors="replace") as f:
        tmv_data = yaml.safe_load(f)

    timeline = tmv_data.get("Timeline", {})
    tracks = timeline.get("Tracks", [])
    bag_start_time = hdf["/metadata"].attrs["bag_start_time"]

    if "labels" not in hdf:
        labels_root = hdf.create_group("labels")
    else:
        labels_root = hdf["labels"]

    for track in tracks:
        tmv_label = track.get("Label")
        if tmv_label is None:
            continue

        if str(tmv_label) not in mapping:
            print(f"[TMV] Skipping unmapped TMV label: {tmv_label}")
            continue

        tracking_id = mapping[str(tmv_label)]
        branches = track.get("Branches", [])
        start_times, durations, actions = [], [], []

        for branch in branches:
            for span in branch.get("Spans", []):
                start = span.get("Start")
                duration = span.get("Duration")
                label_name = span.get("Label")
                if start is not None and duration is not None and label_name:
                    start_times.append(start)
                    durations.append(duration)
                    actions.append(label_name)

        if not start_times:
            print(f"[TMV] No spans found for TMV label {tmv_label}, skipping.")
            continue

        group_path = f"labels/{tracking_id}"
        if group_path in hdf:
            del hdf[group_path]

        label_group = labels_root.create_group(str(tracking_id))
        absolute_start_times = [s + bag_start_time for s in start_times]

        label_group.create_dataset("start_times", data=np.array(absolute_start_times, dtype="f8"))
        label_group.create_dataset("durations", data=np.array(durations, dtype="f8"))
        label_group.create_dataset("actions", data=np.array(actions, dtype=h5py.string_dtype(encoding="utf-8")))

        print(f"[TMV] Inserted labels for ID {tracking_id} ({len(start_times)} spans).")

    print("[TMV] Label insertion complete.")

# Plan for label_extractor.py
# 1. Read TMV YAML safely (yaml.safe_load).
# 2. Loop through Timeline → Tracks.
# 3. Skip tracks whose label not in mapping.
# 4. Flatten all spans from all branches.
# 5. Compute absolute_start_times = [start + bag_start_time].
# 6. Write to /labels/<tracking_id>/:
# - start_times (float64)
# - durations (float64)
# - actions (string)
# 7. If the group already exists, delete it before writing fresh.