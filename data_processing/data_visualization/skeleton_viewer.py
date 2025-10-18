import argparse
import h5py
from pathlib import Path
from data_processing.data_visualization.skeleton_visualizer import SkeletonVisualizer
from data_processing.data_visualization.label_visualizer import LabelVisualizer
from data_processing.data_visualization.environment_visualizer import EnvironmentVisualizer
from data_processing.data_visualization.sensor_visualizer import SensorVisualizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize skeletons from extracted HDF5.")
    parser.add_argument("--file", required=True, help="Path to extracted .hdf5 file")
    parser.add_argument(
        "--mode",
        choices=["poses_raw", "poses_fused", "poses_stable"],
        default="poses_stable",
        help="Which group to visualize",
    )
    parser.add_argument("--show-joints", action="store_true", help="Show joints (points)")
    parser.add_argument("--show-bones", action="store_true", help="Show bones (lines)")
    parser.add_argument("--show-labels", action="store_true", help="Show synchronized labels")
    parser.add_argument("--show-environment", action="store_true", help="Show environment")
    parser.add_argument("--show-sensors", action="store_true", help="Show camera sensors")
    parser.add_argument("--active-sensors", nargs="*", type=int, default=[], help="List of active sensor IDs")
    parser.add_argument("--selected-sensors", nargs="*", type=int, default=[], help="List of selected sensor IDs")

    args = parser.parse_args()

    print(f"[INFO] Visualizing group: /{args.mode}")

    with h5py.File(Path(args.file), "r") as f:
        # ------------------ Visualize skeletons ------------------
        viz = SkeletonVisualizer(show_joints=args.show_joints, show_bones=args.show_bones)
        viz.visualize_dataset(f, args.mode)

        # ------------------ Visualize labels ------------------
        if args.show_labels:
            label_viz = LabelVisualizer(show_labels=True)
            label_viz.visualize_all(f, args.mode)
        
        # ------------------ Visualize environment ------------------
        if args.show_environment:
            env_viz = EnvironmentVisualizer(show_environment=True)
            env_viz.visualize_all_objects(f)

        # ------------------ Visualize sensors ------------------
        if args.show_sensors:
            sensor_viz = SensorVisualizer(show_sensors=True)
            active = set(args.active_sensors or [])
            selected = set(args.selected_sensors or [])
            sensor_viz.visualize_sensors(f, active_sensors=active, selected_sensors=selected)
            # Link into skeleton visualizer for dynamic updates
            viz.sensor_viz = sensor_viz
            sensor_viz.hdf = f


# Plan for skeleton_viewer.py

# 1. Load an `.hdf5` file.
# 2. Let you choose between `/poses_raw` or `/poses_fused`.
# 3. For each `tracking_id`, read joint positions and timestamps.
# 4. Use **rerun** to visualize skeletons (points and lines) in sequence.
# 5. Use `common_utils.azure_kinect_skeleton.BONES` for bone connections.