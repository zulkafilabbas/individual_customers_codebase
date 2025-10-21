# import rerun as rr
# import h5py
# import numpy as np
# from data_processing.data_extraction.stable_selector import StableSelector
# from data_processing.data_visualization.sensor_visualizer import SensorVisualizer

# rr.init("Stable Selector Body Geometry Test", spawn=True)

# selector = StableSelector()
# sensor_viz = SensorVisualizer(show_sensors=True)

# path = r"C:\Users\zulkafil\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5"

# with h5py.File(path, "r") as f:
#     tid = list(f["poses_fused"].keys())[0]
#     joints = f[f"poses_fused/{tid}/joints"][0]

#     # detect which sensors actually have detections for this TID
#     active_sources = []
#     for s in f[f"poses_raw/{tid}"]:
#         src_group = f[f"poses_raw/{tid}/{s}"]
#         for track_name in src_group:
#             if "joints" in src_group[track_name]:
#                 active_sources.append(s)
#                 break  # stop after first valid track
#     active_sensor_ids = {int(src.split("_")[1]) for src in active_sources}

#     sensor_viz.visualize_sensors(f, active_sensors=active_sensor_ids, selected_sensors=set())
#     selector.draw_body_geometry(rr, joints, thickness=0.08)

#     # include environment cuboids once
#     selector.draw_environment_cuboids(rr, color=(180, 180, 180))

#     # iterate through only sensors that actually have detections
#     for src in active_sources:
#         sid = int(src.split("_")[1])
#         # DEBUG: List available keys under this sensor group
#         print(f"\n=== Listing available keys under sensors/2025/sensor_windows_{sid:02d} ===")
#         for k in f[f"sensors/2025/sensor_windows_{sid:02d}"].keys():
#             print(" -", k)
#         # Use 'xyz' as the true 3D position if available
#         sensor_path = f"sensors/2025/sensor_windows_{sid:02d}/xyz"
#         sensor_pos = f[sensor_path][()]
#         print(f"[Sensor {sid}] position from 'xyz':", sensor_pos)

#         # use this sensor’s own detection joints
#         # Find the first valid track with "joints" present (same logic as above)
#         src_group = f[f"poses_raw/{tid}/{src}"]
#         found_src_joints = False
#         for track_name in src_group:
#             if "joints" in src_group[track_name]:
#                 src_joints = src_group[track_name]["joints"][0]
#                 found_src_joints = True
#                 break
#         if not found_src_joints:
#             continue
#         print(f"Example joint for Sensor {sid}:", src_joints[0])

#         print(f"[RayTest] Drawing rays for sensor {sid}...")
#         cuboids = selector.build_body_cuboids(joints, thickness=0.08)
#         cuboids += selector.environment  # append converted OBBs

#         all_points = []
#         all_hit_points = []
#         rays = []

#         for joint_idx, joint in enumerate(src_joints):
#             points = selector.get_points_on_sphere(joint, radius=0.03, n=50)
#             all_points.extend(points)

#             for p in points:
#                 ray_dir = p - sensor_pos
#                 ray_len = np.linalg.norm(ray_dir)
#                 ray_dir /= ray_len

#                 closest_hit = None
#                 min_dist = ray_len

#                 for c in cuboids:
#                     hit, dist = selector.ray_intersects_cuboid(sensor_pos, ray_dir, c)
#                     if hit and 0.0 < dist < min_dist:
#                         min_dist = dist
#                         closest_hit = c

#                 if closest_hit is not None:
#                     end_point = sensor_pos + ray_dir * min_dist
#                 else:
#                     end_point = p

#                 rays.append(end_point)
#                 if closest_hit is None:
#                     all_hit_points.append(p)

#         # draw rays
#         from rerun import LineStrips3D
#         ray_colors = []
#         for end_point in rays:
#             if len(all_hit_points) > 0 and np.any(np.all(np.isclose(end_point, all_hit_points, atol=1e-6), axis=1)):
#                 ray_colors.append([0, 255, 0, 255])  # unoccluded
#             else:
#                 ray_colors.append([200, 10, 10, 180])  # occluded

#         origins = np.repeat(sensor_pos[None, :], len(rays), axis=0)
#         rr.log(
#             f"sensor_{sid:02d}/sensor_rays",
#             LineStrips3D(
#                 np.stack([origins, np.array(rays)], axis=1),
#                 colors=ray_colors,
#                 radii=0.0005,
#             ),
#         )

#         if all_hit_points:
#             rr.log(f"sensor_{sid:02d}/intersections",
#                    rr.Points3D(all_hit_points, colors=[[0, 255, 0]], radii=0.007))
