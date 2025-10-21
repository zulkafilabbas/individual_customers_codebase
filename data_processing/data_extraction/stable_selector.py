#!/usr/bin/env python3
"""
stable_selector.py
--------------------------------
Selects one skeleton from the set of raw poses available for each track (tid) at each timestamp. 
The selected skeleton is the one that minimizes the cost function.

The cost function is a weighted sum of the following components:
- continuity  - prefer same sensor as previous frame
- distance    - closer sensors preferred
- line of sight - person is in line of sight of the sensor or occluded by an object or themselves
- confidence  - "how good" is the detection according to the azure kinect sdk (track_states)

Logic:
- Reads all raw detections under /poses_raw/<tid>/source_<sid>.
- For each frame, evaluates a simple cost per (tid, sid):
    continuity  - prefer same sensor as previous frame
    distance    - closer sensors preferred
    line of sight - person is in line of sight of the sensor or occluded by an object or themselves
    confidence  - "how good" is the detection according to the azure kinect sdk (track_states)
- Keeps the best sensor as long as it remains active.
- Writes final "stable" skeleton sequence to /poses_stable/<tid>, with timestamps, joints, and track_states, identical structure to /poses_fused/<tid>, just more stable.
--------------------------------
"""

import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R
from common_utils.loader import JsonLoader

class StableSelector:
    def __init__(self, extrinsics_year="2025"):
        loader = JsonLoader("common_utils")
        self.extrinsics = loader.get_extrinsics(extrinsics_year)
        self.skeleton = loader.get_skeleton()
        self.joints = self.skeleton["joints"]
        self.bones = self.skeleton["bones"]
        self.environment = loader.get_environment()
        self.convert_environment_objects()

    def get_points_on_sphere(self, center, radius=0.1, n=100):
        """Return n points roughly evenly distributed on a sphere surface (Fibonacci lattice)."""
        # Fibonacci lattice method for approximately uniform sampling
        indices = np.arange(0, n)
        phi = 2 * np.pi * indices / ((1 + np.sqrt(5)) / 2)
        theta = np.arccos(1 - 2*(indices + 0.5) / n)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.stack([x, y, z], axis=1) + center

    def draw_joint_sphere(self, rr, joint_center, radius=0.1, n=100, name="joint_sphere"):
        pts = self.get_points_on_sphere(joint_center, radius, n)
        rr.log(name, rr.Points3D(pts, radii=0.005))

    
    def get_bone_vectors(self, joints):
        """Return bone vectors for each bone in the skeleton."""
        bone_vectors = []
        for bone in self.bones:
            child_joint = joints[bone[0]]
            parent_joint = joints[bone[1]]
            bone_vector = child_joint - parent_joint
            bone_vectors.append(bone_vector)
        return bone_vectors

    def build_body_cuboids(self, joints, thickness=0.04, shrink=0.85):
        """
        Build simplified body geometry with ~12 blocks.
        Torso sides use 4-joint regions; limbs use 2-joint cuboids.
        """
        cuboids = []

        # --- Per-axis shrink factors for each region ---
        # length, width, height
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
            "head_block": (1.1, 1.1, 1.4),  # nose, left_eye, right_eye, head(back)
        }

        # --- Region definitions (joint indices) ---
        regions = [
            ("right_torso", [12, 11, 0, 22]),  # shoulder, clavicle, pelvis, hip
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
            ("head_block", [29, 31]),  # left_ear, right_ear indices
        ]

        for name, joint_idxs in regions:
            # 2. Retrieve correct per-axis shrink tuple (default to 1,1,1)
            sx, sy, sz = region_shrink.get(name, (1.0, 1.0, 1.0))

            if len(joint_idxs) == 2:
                # --- Standard bone cuboid (2-joint case) ---
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
                # 3. Apply three shrink factors to cuboid dims
                dims = np.array([thickness * sx, thickness * sy, effective_length * sz])
                cuboids.append({"name": name, "center": center, "dims": dims, "rotation": rot})

            elif len(joint_idxs) == 4:
                # --- Approximate torso polygon as oriented box ---
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
                # 3. Apply three shrink factors to torso cuboid dims
                dims = np.array([width * sx, height * sy, thickness * 2 * sz])
                cuboids.append({"name": name, "center": center, "dims": dims, "rotation": rot})

        return cuboids


    def draw_body_geometry(self, rr, joints, thickness=0.03, shrink=0.5):
        """Visualize skeleton bones as cuboids using rerun Boxes3D."""
        from scipy.spatial.transform import Rotation
        cuboids = self.build_body_cuboids(joints, thickness, shrink)
        for i, c in enumerate(cuboids):
            quat = Rotation.from_matrix(c["rotation"]).as_quat()  # returns [x, y, z, w]
            rr.log(
                f"body_cuboid/{i}",
                rr.Boxes3D(
                    centers=[c["center"]],
                    half_sizes=[c["dims"] / 2.0],
                    quaternions=[quat],
                    colors=[(150, 200, 255)],
                ),
            )

    def draw_sensor_rays(self, rr, sensor_pos, sampled_points, color=(255, 0, 0), thickness=0.0005):
        """Draw rays from a single sensor position to each sampled point, with configurable thickness."""
        origins = np.repeat(sensor_pos[None, :], len(sampled_points), axis=0)
        rr.log(
            "sensor_rays",
            rr.LineStrips3D(
                np.stack([origins, sampled_points], axis=1),
                colors=[color],
                radii=thickness,
            ),
        )

    def ray_intersects_cuboid(self, ray_origin, ray_dir, cuboid, eps=1e-6):
        """
        Ray–OBB intersection that also returns intersection distance.
        Returns (hit, tmin)
        """
        center = cuboid["center"]
        half_sizes = cuboid["dims"] / 2.0
        R = cuboid["rotation"]

        # Transform to cuboid local space
        inv_R = R.T
        local_origin = inv_R @ (ray_origin - center)
        local_dir = inv_R @ ray_dir

        tmin, tmax = -np.inf, np.inf
        for i in range(3):
            if abs(local_dir[i]) < eps:
                if abs(local_origin[i]) > half_sizes[i]:
                    return False, None
            else:
                t1 = (-half_sizes[i] - local_origin[i]) / local_dir[i]
                t2 = ( half_sizes[i] - local_origin[i]) / local_dir[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return False, None

        if tmax < eps:
            return False, None  # behind
        if tmin < eps:
            return False, None  # ray starts inside or too close
        return True, tmin

    def convert_environment_objects(self):
        """Convert 8-vertex cuboids into {center, dims, rotation} format for occlusion."""
        converted = []
        for obj in self.environment.get("objects", []):
            verts = np.array(obj["vertices"])

            # Identify top 4 by max Z and shift them downward
            # Ideally this should be done by the environment extractor or fixed in the environment json but for now this is a quick fix.
            # Specifically sensor 14 was completely blocked due ot a high shelf, same happens with sensor 16!
            # TODO: Fix this in the environment json.
            
            z_max = np.max(verts[:, 2])
            mask = np.isclose(verts[:, 2], z_max, atol=1e-3)
            verts[mask, 2] -= 0.1  # lower top plane by 0.2 m


            if verts.shape != (8, 3):
                continue

            # Compute center as average of all vertices
            center = np.mean(verts, axis=0)

            # Define three local axes from edges
            v1 = verts[1] - verts[0]  # X-axis
            v2 = verts[3] - verts[0]  # Y-axis
            v3 = verts[4] - verts[0]  # Z-axis

            # Dimensions (lengths along local axes)
            dims = np.array([np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)])

            # Normalize to get rotation basis
            x_axis = v1 / np.linalg.norm(v1)
            y_axis = v2 / np.linalg.norm(v2)
            z_axis = v3 / np.linalg.norm(v3)
            rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

            converted.append({
                "name": obj.get("name", "env_object"),
                "center": center,
                "dims": dims,
                "rotation": rotation,
            })
        self.environment = converted # replace raw vertices with {center, dims, rotation} OBBs for occlusion checking

    def draw_environment_cuboids(self, rr, color=(200, 200, 200)):
        """Visualize environment objects as Boxes3D in rerun."""
        from scipy.spatial.transform import Rotation
        import numpy as np
        for i, obj in enumerate(self.environment):
            rotation = obj["rotation"].copy()
            # Ensure right-handed (determinant > 0)
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
