#!/usr/bin/env python3
"""
stable_selector2.py
--------------------------------
Selects one skeleton from the set of raw poses available for each track (tid) across time.  
The goal is to maintain temporal stability — keep using the same sensor as long as its quality remains acceptable, switching only when another sensor clearly performs better.

The cost function is a weighted combination of:
- occlusion  : penalize sensors with low weighted visibility from the OcclusionAnalyzer.
- distance   : penalize sensors that are too far or too close; prefer mid-range detections.
- continuity : prefer sensors that have been stable over the recent time window (e.g., past 10 seconds).

Logic:
- Reads all raw detections under /poses_raw/<tid>/source_<sid>.
- For each timestamp, computes per-sensor cost combining occlusion and distance.
- Tracks the active sensor over a sliding window (~10 s) using timestamps to measure continuity.
- Continues with the current sensor if its metrics remain within tolerance of recent averages.
- Switches only when visibility drops sharply or another sensor's visibility stays better for sustained period of time.
- Overall strategy is to prevent switching between sensors too frequently. This is what we mean by stable selection.
- Writes the chosen stable sequence under /poses_stable/<tid>, matching the structure of /poses_fused/<tid> (timestamps, joints, and track_states).
--------------------------------

"""

import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R
from common_utils.loader import JsonLoader
from data_processing.data_visualization.sensor_visualizer import SensorVisualizer

class StableSelector:
    def __init__(self, extrinsics_year="2025", continuity_window_sec=10.0):
        from data_processing.data_extraction.occlusion_analyzer import OcclusionAnalyzer, Source, Obstacle
        from common_utils.loader import JsonLoader
        loader = JsonLoader("common_utils")
        self.extrinsics = loader.get_extrinsics(extrinsics_year)
        self.environment = loader.get_environment()
        self.skeleton = loader.get_skeleton()
        self.joints = self.skeleton["joints"]
        self.bones = self.skeleton["bones"]
        self.sensors = {}
        for name, pose in self.extrinsics.items():
            if not "xyz" in pose:
                continue
            parts = name.split("_")
            if not parts[-1].isdigit():
                continue
            sid = int(parts[-1])
            self.sensors[sid] = Source(sid, np.array(pose["xyz"]), np.array(pose["ypr"]))

        self.continuity_window_sec = continuity_window_sec
        self.environment = self._convert_environment()
        self.joint_weights = self._build_joint_weights()
        self.sensor_history = []

        # Added: tracks per-sensor cost history for smoothing
        from collections import deque
        self.recent_costs = {}  # sid -> deque(maxlen=15)

    def _smoothed_cost(self, sid, new_cost):
        from collections import deque
        if sid not in self.recent_costs:
            self.recent_costs[sid] = deque(maxlen=60)
        q = self.recent_costs[sid]
        q.append(new_cost)
        return np.mean(q)

    def _convert_environment(self):
        out = []
        for obj in self.environment.get("objects", []):
            verts = np.array(obj["vertices"])
            if verts.shape != (8, 3): continue
            center = np.mean(verts, 0)
            v1, v2, v3 = verts[1]-verts[0], verts[3]-verts[0], verts[4]-verts[0]
            dims = np.array([np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)])
            rot = np.stack([v1/np.linalg.norm(v1), v2/np.linalg.norm(v2), v3/np.linalg.norm(v3)], 1)
            out.append({"center": center, "dims": dims, "rotation": rot})
        return out

    def draw_environment_cuboids(self, rr, color=(200, 200, 200)):
        from scipy.spatial.transform import Rotation
        import numpy as np
        for i, obj in enumerate(self.environment):
            rotation = obj["rotation"].copy()
            if np.linalg.det(rotation) < 0:
                rotation[:, 2] *= -1
            quat = Rotation.from_matrix(rotation).as_quat()
            rr.log(
                f"environment_cuboid/{i}",
                rr.Boxes3D(
                    centers=[obj["center"]],
                    half_sizes=[obj["dims"] / 2.0],
                    quaternions=[quat],
                    colors=[color],
                ),
            )

    def _build_joint_weights(self):
        jw = {jid: 1.0 for jid in self.joints}
        arm = {"SHOULDER_LEFT","SHOULDER_RIGHT","ELBOW_LEFT","ELBOW_RIGHT","WRIST_LEFT","WRIST_RIGHT"}
        front = {"NOSE","EYE_LEFT","EYE_RIGHT","CLAVICLE_LEFT","CLAVICLE_RIGHT","SPINE_CHEST"}
        for jid,(n,_) in self.joints.items():
            if n in front: jw[jid]=3.0
            if n in arm: jw[jid]=10.0
        return jw

    def build_body_cuboids(self, joints, thickness=0.08, shrink=0.85):
        cuboids = []
        region_shrink = {
            "right_torso": (0.9, 0.1, 0.5),
            "left_torso":  (0.9, 0.1, 0.5),
            "right_clavicle_block": (1.0, 1.0, 1.0),
            "left_clavicle_block": (1.0, 1.0, 1.0),
            "right_upper_arm": (0.85, 0.85, 0.85),
            "left_upper_arm": (0.85, 0.85, 0.85),
            "right_lower_arm": (0.8, 0.8, 0.8),
            "left_lower_arm": (0.8, 0.8, 0.8),
            "right_upper_leg": (1.3, 1.3, 0.95),
            "left_upper_leg": (1.3, 1.3, 0.95),
            "right_lower_leg": (1.3, 1.3, 0.95),
            "left_lower_leg": (1.3, 1.3, 0.95),
            "head_block": (1.1, 1.1, 1.4),
        }
        regions = [
            ("right_torso", [12, 11, 0, 22]),
            ("left_torso", [5, 4, 0, 18]),
            ("right_clavicle_block", [11, 12]),
            ("left_clavicle_block", [4, 5]),
            ("right_upper_arm", [12, 13]),
            ("left_upper_arm", [5, 6]),
            ("right_lower_arm", [13, 14]),
            ("left_lower_arm", [6, 7]),
            ("right_upper_leg", [22, 23]),
            ("left_upper_leg", [18, 19]),
            ("right_lower_leg", [23, 24]),
            ("left_lower_leg", [19, 20]),
            ("head_block", [29, 31]),
        ]
        for name, joint_idxs in regions:
            sx, sy, sz = region_shrink.get(name, (1.0, 1.0, 1.0))
            if len(joint_idxs) == 2:
                j1, j2 = joints[joint_idxs]
                axis = j2 - j1
                length = np.linalg.norm(axis)
                if length < 1e-6:
                    continue
                effective_length = length * shrink
                axis_dir = axis / length
                shift = (length - effective_length) * 0.5
                p1 = j1 + axis_dir * shift
                p2 = j2 - axis_dir * shift
                center = (p1 + p2) / 2.0
                axis_unit = (p2 - p1) / np.linalg.norm(p2 - p1)
                z_axis = np.array([0, 0, 1.0])
                if np.allclose(axis_unit, z_axis):
                    rot = np.eye(3)
                else:
                    v = np.cross(z_axis, axis_unit)
                    c = np.dot(z_axis, axis_unit)
                    s = np.linalg.norm(v)
                    vx = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
                    rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
                dims = np.array([thickness * sx, thickness * sy, effective_length * sz])
                cuboids.append({"name": name, "center": center, "dims": dims, "rotation": rot})

            elif len(joint_idxs) == 4:
                pts = joints[joint_idxs]
                center = np.mean(pts, axis=0)
                v1 = pts[0] - pts[2]
                v2 = pts[1] - pts[3]
                width = np.linalg.norm(v1) * 0.5
                height = np.linalg.norm(v2) * 0.5
                rot_axis = np.cross(v1, v2)
                if np.linalg.norm(rot_axis) < 1e-6:
                    rot = np.eye(3)
                else:
                    rot_axis /= np.linalg.norm(rot_axis)
                    z_axis = np.array([0, 0, 1])
                    v = np.cross(z_axis, rot_axis)
                    c = np.dot(z_axis, rot_axis)
                    s = np.linalg.norm(v)
                    vx = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
                    rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
                dims = np.array([width * sx, height * sy, thickness * 2 * sz])
                cuboids.append({"name": name, "center": center, "dims": dims, "rotation": rot})
        return cuboids

    def _compute_visibility(self, sid, joints):
        from data_processing.data_extraction.occlusion_analyzer import OcclusionAnalyzer, Obstacle
        env_obs = [Obstacle(o["center"],o["dims"],o["rotation"],"env") for o in self.environment]
        body_obs = [Obstacle(c["center"],c["dims"],c["rotation"],"self")
                    for c in self.build_body_cuboids(joints,0.08)]
        obs = env_obs + body_obs
        from data_processing.data_extraction.occlusion_analyzer import Source
        analyzer = OcclusionAnalyzer(obs,self.sensors)
        vis = analyzer.analyze_source_frame(self.sensors[sid], joints)
        weights = np.array([self.joint_weights[j] for j in vis.keys()])
        vals = np.array(list(vis.values()))
        return np.average(vals,weights=weights)

    def _compute_distance_penalty(self, sid, joints):
        torso_ids = [0,1,2,3,4,5]
        torso_center = np.mean(joints[torso_ids],0)
        dist = np.linalg.norm(torso_center - self.sensors[sid].position)
        return min(dist/5.0,1.0)

    def _compute_cost(self, sid, joints):
        v = 1.0 - self._compute_visibility(sid,joints)
        d = self._compute_distance_penalty(sid,joints)
        return 5*v + 0.1*d

    def select_and_visualize(self, f, tid, rr):
        rr.init("StableSelectorV2 Visualization", spawn=True)

        from data_processing.data_visualization.sensor_visualizer import SensorVisualizer
        sensor_viz = SensorVisualizer(show_sensors=True)

        # --- Draw environment and sensors ---
        self.draw_environment_cuboids(rr)
        sensor_viz.visualize_sensors(f, selected_sensors={})

        sources = list(f[f"poses_raw/{tid}"].keys())
        source_data = {
            sid: {
                "timestamps": f[f"poses_raw/{tid}/{sid}/0/timestamps"][:],
                "joints": f[f"poses_raw/{tid}/{sid}/0/joints"][:],
            }
            for sid in sources
        }

        all_ts = np.unique(np.concatenate([v["timestamps"] for v in source_data.values()]))
        last_sid = None

        for t in all_ts:
            frame_costs = {}
            for sid in sources:
                ts_array = source_data[sid]["timestamps"]
                idx = np.argmin(np.abs(ts_array - t))
                if abs(ts_array[idx] - t) > 0.05:
                    continue
                joints = source_data[sid]["joints"][idx]
                # Smooth the per-sensor cost (moving average over last 15 frames)
                raw_cost = self._compute_cost(int(sid.split("_")[-1]), joints)
                smoothed_cost = self._smoothed_cost(int(sid.split("_")[-1]), raw_cost)
                frame_costs[sid] = smoothed_cost

            if not frame_costs:
                continue
            best_sid = min(frame_costs, key=frame_costs.get)

            # Stricter continuity logic: higher tolerance + persistence
            if last_sid is not None:
                recent = [s for s in self.sensor_history if t - s[0] <= self.continuity_window_sec]
                if recent and recent[-1][1] == last_sid:
                    curr_cost = frame_costs.get(last_sid, None)
                    if curr_cost is not None:
                        best_cost = min(frame_costs.values())
                        # stay if within 30% or if we have been stable for a few seconds (>=10 frames)
                        if curr_cost < best_cost * 1.3 or len(recent) >= 10:
                            best_sid = last_sid

            self.sensor_history.append((t, best_sid))
            last_sid = best_sid

            # --- Visualization per frame ---
            rr.set_time_seconds("frame", float(t))
            rr.log("chosen_sensor", rr.TextLog(f"Chosen sensor: {best_sid}"))

            # Update sensor visualizer highlighting for the chosen sensor
            sensor_viz.update_sensor_states(selected_sensors={int(best_sid.split('_')[-1])})

            joints = source_data[best_sid]["joints"][np.argmin(
                np.abs(source_data[best_sid]["timestamps"] - t)
            )]

            # Draw the selected skeleton
            rr.log("selected_skeleton/joints",
                rr.Points3D(joints, colors=[(0, 255, 0)], radii=0.015))

            # Draw bones
            lines = np.array([
                [joints[p], joints[c]] for (c, p) in self.bones if np.all(np.isfinite(joints[p])) and np.all(np.isfinite(joints[c]))
            ])
            rr.log("selected_skeleton/bones",
                rr.LineStrips3D(lines, colors=[(0, 255, 0)], radii=0.005))

            rr.log("costs", rr.Scalars(frame_costs))

