# import rerun as rr
# import h5py
# import numpy as np
# from data_processing.data_extraction.stable_selector import StableSelector
# from data_processing.data_visualization.sensor_visualizer import SensorVisualizer

# rr.init("Stable Selector Body Geometry Test", spawn=True)

# selector = StableSelector()
# sensor_viz = SensorVisualizer(show_sensors=True)
# # open any dataset file
# path = r"C:\Users\zulkafil\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5"

# with h5py.File(path, "r") as f:
#     tid = list(f["poses_fused"].keys())[0]
#     joints = f[f"poses_fused/{tid}/joints"][0]

#     sensor_viz.visualize_sensors(f, active_sensors={1,2,3,4,5,6}, selected_sensors={3})
#     selector.draw_body_geometry(rr, joints, thickness=0.08)

#     sensor_pos = f["sensors/2025/sensor_windows_03/xyz"][()]

#     cuboids = selector.build_body_cuboids(joints, thickness=0.08)
#     # include environment cuboids
#     cuboids += selector.environment  # append converted OBBs

#     # For all joints, sample points and check ray intersections
#     all_points = []
#     all_hit_points = []
#     rays = []

#     for joint_idx, joint in enumerate(joints):
#         points = selector.get_points_on_sphere(joint, radius=0.03, n=50)
#         all_points.extend(points)

#         for p in points:
#             ray_dir = p - sensor_pos
#             ray_len = np.linalg.norm(ray_dir)
#             ray_dir /= ray_len

#             closest_hit = None
#             min_dist = ray_len  # max distance = sphere surface

#             for c in cuboids:
#                 hit, dist = selector.ray_intersects_cuboid(sensor_pos, ray_dir, c)
#                 if hit and 0.0 < dist < min_dist:
#                     min_dist = dist
#                     closest_hit = c

#             if closest_hit is not None:
#                 blocked = True
#                 end_point = sensor_pos + ray_dir * min_dist  # stop at hit
#             else:
#                 blocked = False
#                 end_point = p

#             rays.append(end_point)
#             if not blocked:
#                 all_hit_points.append(p)

#     # Draw all rays
#     # selector.draw_sensor_rays(rr, sensor_pos, np.array(all_points), color=[255, 0, 0])

#     # Draw all rays but the ones that hit some occluding object should truncate at that point. Others are drawn to the sample points on the sphere.
#     # Draw rays with red color and alpha 120 if occluded, alpha 255 if not occluded
#     ray_colors = []
#     for end_point in rays:
#         # If the ray endpoint is in all_hit_points, we consider it unoccluded (made it to the sphere)
#         if np.any(np.all(np.isclose(end_point, all_hit_points, atol=1e-6), axis=1)):
#             ray_colors.append([0, 255, 0, 255])  # Not occluded: alpha 255
#         else:
#             ray_colors.append([200, 10, 10, 180])  # Occluded: alpha 120

#     # Use the overridden color per ray
#     from rerun import LineStrips3D
#     origins = np.repeat(sensor_pos[None, :], len(rays), axis=0)
#     rr.log(
#         "sensor_rays_alpha",
#         LineStrips3D(
#             np.stack([origins, np.array(rays)], axis=1),
#             colors=ray_colors,
#             radii=0.0005,
#         ),
#     )

#     # Highlight only the hit points in green
#     if all_hit_points:
#         rr.log("intersections", rr.Points3D(all_hit_points, colors=[[0,255,0]], radii=0.007))
    

#     selector.draw_environment_cuboids(rr, color=(180, 180, 180))