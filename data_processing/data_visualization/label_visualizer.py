import rerun as rr
import numpy as np
from data_processing.data_extraction.label_parser import read_labels_for_id

class LabelVisualizer:
    """
    Visualizes time-synchronized TMV labels for each tracking ID.
    Displays only the label(s) active at the current timestamp.
    """

    def __init__(self, show_labels=True):
        self.show_labels = show_labels

    def _active_labels(self, label_spans, ts):
        """Return all labels active at timestamp ts."""
        active = []
        for start, dur, label in label_spans:
            if start <= ts <= (start + dur):
                active.append(label)
        return active

    def visualize_labels_for_tid(self, hdf, tid, timestamps, base_path):
        """
        Log time-synchronized labels for one tracking ID.
        base_path = e.g. 'poses_fused/12' or 'poses_raw/12/source_0'
        """
        label_spans = read_labels_for_id(hdf, tid)
        if not label_spans:
            return

        for ts in timestamps:
            active = self._active_labels(label_spans, ts)
            if not active:
                continue

            rr.set_time_seconds("frame", float(ts))
            for l in active:
                if l.location != "None":
                    rr.log(
                        f"{base_path}/labels/location",
                        rr.TextLog(f"Location: {l.location}", color=[120, 180, 255]),  # soft blue
                    )
                if l.arm_action != "None":
                    rr.log(
                        f"{base_path}/labels/arm_action",
                        rr.TextLog(f"Arm Action: {l.arm_action}", color=[255, 150, 120]),  # warm red
                    )
                if l.interaction != "None":
                    rr.log(
                        f"{base_path}/labels/interaction",
                        rr.TextLog(f"Interaction: {l.interaction}", color=[150, 255, 150]),  # light green
                    )


    def visualize_all(self, hdf, mode="poses_raw"):
        """Attach label visualization to all skeletons under a given mode."""
        if not self.show_labels or "labels" not in hdf:
            print("[LabelVisualizer] No labels to show.")
            return

        label_ids = list(hdf["labels"].keys())
        print(f"[LabelVisualizer] Found {len(label_ids)} labeled tracks.")

        top_group = hdf[mode]
        for tid in label_ids:
            tid_int = int(tid)
            if str(tid_int) not in top_group:
                continue

            if mode == "poses_raw":
                for src in top_group[str(tid_int)].keys():
                    inner = top_group[str(tid_int)][src][str(tid_int)]
                    timestamps = inner["timestamps"][:]
                    self.visualize_labels_for_tid(hdf, tid_int, timestamps, f"{mode}/{tid_int}/{src}")
            else:
                tid_group = top_group[str(tid_int)]
                timestamps = tid_group["timestamps"][:]
                self.visualize_labels_for_tid(hdf, tid_int, timestamps, f"{mode}/{tid_int}")
