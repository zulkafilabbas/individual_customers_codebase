import rerun as rr
import numpy as np
from common_utils.loader import JsonLoader
import itertools
import hashlib

# Unique color generator based on mode, tracking_id, and optional source_id
def color_from_identity(mode, tracking_id, source_id=None, brightness=0.6):
    """
    Generate a persistent, lighter deterministic RGBA color (0–1) 
    from mode, tracking_id, and optional source_id.
    `brightness` controls how much to mix with white (0–1).
    """
    key = f"{mode}_{tracking_id}"
    if source_id is not None:
        key += f"_{source_id}"

    digest = hashlib.md5(key.encode()).hexdigest()
    h = int(digest[:6], 16)
    r = (h >> 16) & 255
    g = (h >> 8) & 255
    b = h & 255

    # Normalize and mix with white to lighten the color
    base = np.array([r, g, b]) / 255.0
    white = np.ones(3)
    light_rgb = base * (1 - brightness) + white * brightness

    if mode == "poses_raw":
        alpha = 0.3
    elif mode == "poses_fused":
        alpha = 0.6
    elif mode == "poses_stable":
        alpha = 0.9
    else:
        alpha = 1.0

    return (*light_rgb, alpha)


class SkeletonVisualizer:
    def __init__(self, show_joints=True, show_bones=True):
        self.show_joints = show_joints
        self.show_bones = show_bones
        self.loader = JsonLoader("common_utils")
        self.skeleton_def = self.loader.get_skeleton()

    def _draw_frame(self, group_path, joints, timestamp, color):
        """Draw one frame (optionally points and bones)."""
        rr.set_time_seconds("frame", float(timestamp))

        if self.show_joints:
            rr.log(f"{group_path}/joints", rr.Points3D(joints, colors=color, radii=0.015))

        if self.show_bones:
            bones = np.array([(joints[a], joints[b]) for a, b in self.skeleton_def["bones"]])
            rr.log(f"{group_path}/bones", rr.LineStrips3D(bones, colors=color))

    def visualize_group(self, group_path, group, color):
        """Render all frames for a given tracking_id group."""
        timestamps = group["timestamps"][:]
        joints = group["joints"][:]
        for t, frame in zip(timestamps, joints):
            self._draw_frame(group_path, frame, t, color)

    def visualize_dataset(self, hdf, mode="poses_raw"):
        """Render all skeletons under a chosen mode (raw/fused/stable)."""
        if mode not in hdf:
            print(f"Group '{mode}' not found in file.")
            return

        rr.init(f"Skeleton Viewer: {mode}", spawn=True)
        top_group = hdf[mode]

        for tid in top_group.keys():
            tid_group = top_group[tid]

            if mode == "poses_raw":
                for src in tid_group.keys():
                    inner = tid_group[src][tid]
                    color = color_from_identity(mode, tid, src)
                    self.visualize_group(f"{mode}/{tid}/{src}", inner, color)
            else:
                color = color_from_identity(mode, tid)
                self.visualize_group(f"{mode}/{tid}", tid_group, color)