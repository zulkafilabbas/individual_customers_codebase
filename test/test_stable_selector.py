import rerun as rr
import h5py
import numpy as np
from data_processing.data_extraction.stable_selector import StableSelector
from data_processing.data_visualization.sensor_visualizer import SensorVisualizer

rr.init("Stable Selector Body Geometry Test", spawn=True)

selector = StableSelector()
sensor_viz = SensorVisualizer(show_sensors=True)
# open any dataset file
path = r"C:\Users\zulkafil\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5"

# with h5py.File(path, "r") as f:
#     tid = list(f["poses_fused"].keys())[0]
#     joints = f[f"poses_fused/{tid}/joints"][0]  # first frame

#     sensor_viz.visualize_sensors(f, active_sensors={1,2,3,4,5,6}, selected_sensors={3})
#     selector.draw_body_geometry(rr, joints, thickness=0.08)

# # sample points on all joints
# for idx, joint in enumerate(joints):
#     selector.draw_joint_sphere(rr, joint, radius=0.03, n=200, name=f"joint_sphere/{idx}")

# sensor_pos = f["sensors/2025/sensor_windows_13/position"][()]
# head_idx = 26
# points = selector.get_points_on_sphere(joints[head_idx], radius=0.03, n=50)
# selector.draw_sensor_rays(rr, sensor_pos, points)


with h5py.File(path, "r") as f:
    tid = list(f["poses_fused"].keys())[0]
    joints = f[f"poses_fused/{tid}/joints"][0]

    sensor_viz.visualize_sensors(f, active_sensors={1,2,3,4,5,6}, selected_sensors={3})
    selector.draw_body_geometry(rr, joints, thickness=0.08)

    sensor_pos = f["sensors/2025/sensor_windows_07/xyz"][()]

    cuboids = selector.build_body_cuboids(joints, thickness=0.08)

    # For all joints, sample points and check ray intersections
    all_points = []
    all_hit_points = []

    for joint_idx, joint in enumerate(joints):
        points = selector.get_points_on_sphere(joint, radius=0.03, n=50)
        all_points.extend(points)

        for p in points:
            ray_dir = p - sensor_pos
            ray_dir /= np.linalg.norm(ray_dir)

            blocked = False
            for c in cuboids:
                hit, dist = selector.ray_intersects_cuboid(sensor_pos, ray_dir, c)
                if hit and 0.0 < dist < np.linalg.norm(p - sensor_pos):
                    blocked = True
                    break

            if not blocked:
                all_hit_points.append(p)

    # Draw all rays
    selector.draw_sensor_rays(rr, sensor_pos, np.array(all_points), color=[255, 0, 0])
    # Highlight only the hit points in green
    if all_hit_points:
        rr.log("intersections", rr.Points3D(all_hit_points, colors=[[0,255,0]], radii=0.007))

