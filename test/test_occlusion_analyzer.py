#!/usr/bin/env python3
import rerun as rr
import h5py
import numpy as np

from data_processing.data_extraction.stable_selector import StableSelector
from data_processing.data_visualization.sensor_visualizer import SensorVisualizer
from data_processing.data_extraction.occlusion_analyzer import (
    Ray, Obstacle, JointSphere, RayBundle, OcclusionAnalyzer, OcclusionVisualizer, Source
)

# --------------------------------------------
# Configurable parameters
# --------------------------------------------
N_SAMPLES = 10
# MAX_FRAMES = 5   # limit for debugging; set None for all frames
MAX_FRAMES = None # all frames

# --------------------------------------------
# Setup rerun + helpers
# --------------------------------------------
rr.init("Occlusion Analyzer Multi-Frame", spawn=True)

selector = StableSelector()
sensor_viz = SensorVisualizer(show_sensors=True)

path = r"C:\Users\zulkafil\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5"

# --------------------------------------------
# Load dataset and setup
# --------------------------------------------
with h5py.File(path, "r") as f:
    tid = list(f["poses_fused"].keys())[0]
    fused_joints_all = f[f"poses_fused/{tid}/joints"]
    num_frames = fused_joints_all.shape[0]
    if MAX_FRAMES:
        num_frames = min(num_frames, MAX_FRAMES)

    # find active sources
    active_sources = []
    for s in f[f"poses_raw/{tid}"]:
        src_group = f[f"poses_raw/{tid}/{s}"]
        for track_name in src_group:
            if "joints" in src_group[track_name]:
                active_sources.append(s)
                break
    active_sensor_ids = {int(src.split("_")[1]) for src in active_sources}

    # setup sensors
    sensors = {}
    for src in active_sources:
        sid = int(src.split("_")[1])
        base_path = f"sensors/2025/sensor_windows_{sid:02d}"
        sensor_pos = f[f"{base_path}/xyz"][()]
        ypr = f[f"{base_path}/ypr"][()] if "ypr" in f[base_path] else None
        sensors[sid] = Source(sid, sensor_pos, ypr)

    # environment setup
    env_obstacles = [
        Obstacle(e["center"], e["dims"], e["rotation"], category="env")
        for e in selector.environment
    ]

    # --------------------------------------------
    # Global visualization setup
    # --------------------------------------------
    sensor_viz.visualize_sensors(f, active_sensors=active_sensor_ids, selected_sensors=set())
    viz = OcclusionVisualizer()
    viz.draw_obstacles(env_obstacles, prefix="environment")

    # --------------------------------------------
    # Loop through frames
    # --------------------------------------------
    for frame_idx in range(num_frames):
        print(f"\n=== Frame {frame_idx} ===")

        frame_prefix = f"frame_{frame_idx:04d}"
        fused_joints = fused_joints_all[frame_idx]
        viz.draw_joints(fused_joints, name=f"{frame_prefix}/fused_joints")

        for src in active_sources:
            sid = int(src.split("_")[1])
            sensor_pos = sensors[sid].position

            src_group = f[f"poses_raw/{tid}/source_{sid}"]
            src_joints = None
            for track_name in src_group:
                if "joints" in src_group[track_name]:
                    num_src_frames = src_group[track_name]["joints"].shape[0]
                    if frame_idx >= num_src_frames:
                        continue  # skip this source for this frame
                    src_joints = src_group[track_name]["joints"][frame_idx]
                    break
            if src_joints is None:
                continue

            # per-detection obstacles
            body_obstacles = [
                Obstacle(c["center"], c["dims"], c["rotation"], category="self")
                for c in selector.build_body_cuboids(src_joints, thickness=0.08)
            ]
            obstacles_for_this_detection = env_obstacles + body_obstacles
            analyzer = OcclusionAnalyzer(obstacles_for_this_detection, sensors)

            vis_per_joint = analyzer.analyze_source_frame(sensors[sid], src_joints, n_samples=N_SAMPLES)
            print(f"Sensor {sid} mean visibility: {np.mean(list(vis_per_joint.values())):.2f}")

            # --- Visualization per frame ---
            viz.draw_obstacles(body_obstacles, prefix=f"{frame_prefix}/source_{sid}/body")
            viz.draw_joints(src_joints, color=(0,128,255), radii=0.015,
                            name=f"{frame_prefix}/source_{sid}/detected_joints")

            for j_idx, joint in enumerate(src_joints):
                js = JointSphere(joint)
                js.sample_points(N_SAMPLES)
                bundle = RayBundle(sensor_id=sid, frame_idx=j_idx)
                for sample_point in js.samples:
                    bundle.rays.append(Ray(sensor_pos, sample_point))
                bundle.cast_against(obstacles_for_this_detection)
                viz.draw_bundle(bundle, name=f"{frame_prefix}/source_{sid}/joint_{j_idx:02d}/rays")
                viz.draw_joints(js.samples, radii=0.005, color=(255,255,0),
                                name=f"{frame_prefix}/source_{sid}/joint_{j_idx:02d}/samples")
