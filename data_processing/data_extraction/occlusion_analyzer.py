#!/usr/bin/env python3
"""
occlusion_analyzer.py
----------------------------------------
Handles geometric visibility and occlusion analysis between sensors and skeletons.

Core concepts:
- A Ray represents a single line from a sensor toward a sampled point on a joint sphere.
- Obstacles (Cuboids) represent environment or body parts that may block rays.
- RayBundle holds all rays from a single sensor for one skeleton frame.
- JointSphere samples multiple points around each joint for occlusion testing.
- OcclusionAnalyzer orchestrates the per-frame visibility computation.

This module is standalone and can be imported by StableSelector or any other system
requiring occlusion-based reliability estimation.
----------------------------------------
"""

import numpy as np
from typing import List, Optional, Dict

class Source:
    """Represents one physical sensor (source) with position and orientation."""
    def __init__(self, sensor_id: int, position: np.ndarray, ypr: Optional[np.ndarray] = None):
        self.sensor_id = sensor_id
        self.position = np.asarray(position, dtype=np.float32)
        self.ypr = np.asarray(ypr, dtype=np.float32) if ypr is not None else np.zeros(3, dtype=np.float32)

    def to_dict(self):
        return {"id": self.sensor_id, "position": self.position, "ypr": self.ypr}

# ============================================================
# --- Core geometric primitives
# ============================================================

class Ray:
    """Represents a single ray from sensor → point in 3D space, with a defined endpoint."""

    def __init__(self, origin: np.ndarray, end_point: np.ndarray):
        self.origin = np.asarray(origin, dtype=np.float32)
        self.end_point_target = np.asarray(end_point, dtype=np.float32)
        diff = self.end_point_target - self.origin
        self.length = np.linalg.norm(diff)
        if self.length < 1e-8:
            raise ValueError("Ray has zero length.")
        self.direction = diff / self.length
        self.hit_distance: Optional[float] = None
        self.hit_object: Optional["Obstacle"] = None  # Reserved for compatibility

    def intersect(self, cuboid: "Obstacle") -> Optional[float]:
        """
        Compute intersection distance between this ray and an oriented cuboid (OBB).
        Returns the smallest positive distance or None if no intersection.
        """
        center, dims, R = cuboid.center, cuboid.dims / 2.0, cuboid.rotation

        # Transform ray into cuboid's local space
        inv_R = R.T
        local_origin = inv_R @ (self.origin - center)
        local_dir = inv_R @ self.direction

        tmin, tmax = -np.inf, np.inf
        eps = 1e-6
        for i in range(3):
            if abs(local_dir[i]) < eps:
                # Ray is parallel to this slab
                if abs(local_origin[i]) > dims[i]:
                    return None
            else:
                t1 = (-dims[i] - local_origin[i]) / local_dir[i]
                t2 = ( dims[i] - local_origin[i]) / local_dir[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return None

        if tmax < 0:
            return None  # intersection behind origin
        if tmin < 0:
            tmin = tmax  # ray starts inside, exit point
        return tmin if (tmin > 0 and tmin < self.length) else None

    def shorten(self, distance: float):
        """Truncate ray to intersection distance."""
        self.hit_distance = distance

    def end_point(self) -> np.ndarray:
        """Return ray endpoint (either to the target or stopped by intersection)."""
        if self.hit_distance is None:
            return self.origin + self.direction * self.length
        return self.origin + self.direction * self.hit_distance


# ============================================================
# --- Obstacle / Cuboid
# ============================================================

class Obstacle:
    """Represents a cuboid that may occlude rays."""

    def __init__(self, center: np.ndarray, dims: np.ndarray, rotation: np.ndarray, category: str = "env"):
        self.center = np.asarray(center, dtype=np.float32)
        self.dims = np.asarray(dims, dtype=np.float32)
        self.rotation = np.asarray(rotation, dtype=np.float32)
        self.category = category

    def intersect(self, ray: Ray) -> Optional[float]:
        """Compute intersection distance between this cuboid and a Ray (OBB intersection)."""
        raise NotImplementedError


# ============================================================
# --- JointSphere
# ============================================================

class JointSphere:
    """Generates sample points on a sphere around a joint."""

    def __init__(self, center: np.ndarray, radius: float = 0.03):
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = float(radius)
        self.samples: Optional[np.ndarray] = None  # (N, 3)

    def sample_points(self, n: int = 100) -> np.ndarray:
        """Generate approximately uniform points on the sphere using Fibonacci lattice."""
        indices = np.arange(0, n, dtype=np.float32)
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        y = 1 - (indices / float(n - 1)) * 2  # from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi * indices
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points = np.stack([x, y, z], axis=1) * self.radius + self.center
        self.samples = points
        return points

    def directions_from(self, sensor_pos: np.ndarray) -> np.ndarray:
        """Compute normalized directions from sensor to each sampled point."""
        if self.samples is None:
            raise ValueError("Call sample_points() before computing directions.")
        dirs = self.samples - sensor_pos
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8
        return dirs / norms

# ============================================================
# --- RayBundle
# ============================================================

class RayBundle:
    """Collection of rays from one sensor to all sampled joint-sphere points."""

    def __init__(self, sensor_id: int, frame_idx: int):
        self.sensor_id = sensor_id
        self.frame_idx = frame_idx
        self.rays: List[Ray] = []
        self._hit_mask = None  # store results for vectorized checks

    def cast_against(self, obstacles: List[Obstacle]) -> None:
        """
        Check all rays for intersection against given obstacles.
        Updates each ray’s hit_distance to the closest hit.
        """
        if not self.rays or not obstacles:
            return

        origins = np.array([r.origin for r in self.rays])
        dirs = np.array([r.direction for r in self.rays])
        ray_lengths = np.array([r.length for r in self.rays], dtype=np.float32)
        M = len(self.rays)
        min_dists = np.full(M, np.inf, dtype=np.float32)

        for obs in obstacles:
            center, dims, R = obs.center, obs.dims / 2.0, obs.rotation
            inv_R = R.T
            ro_local = (origins - center) @ inv_R
            rd_local = dirs @ inv_R

            inv_dir = np.where(np.abs(rd_local) < 1e-8, 1e8, 1.0 / rd_local)
            tmin = (-dims - ro_local) * inv_dir
            tmax = ( dims - ro_local) * inv_dir
            t1 = np.minimum(tmin, tmax)
            t2 = np.maximum(tmin, tmax)

            t_near = np.max(t1, axis=1)
            t_far = np.min(t2, axis=1)
            valid_hits = (t_near <= t_far) & (t_far > 0.0)
            dists = np.where(valid_hits, np.maximum(t_near, 0.0), np.inf)
            # Clamp intersection distance to the ray's length - anything past endpoint is ignored!
            dists = np.minimum(dists, ray_lengths)
            closer = dists < min_dists
            min_dists = np.where(closer, dists, min_dists)

        # min_dists are now either inf (no hit) or the nearest hit, but never farther than the ray's intended length
        self._hit_mask = np.isfinite(min_dists) & (min_dists < ray_lengths)
        for i, r in enumerate(self.rays):
            if np.isfinite(min_dists[i]) and min_dists[i] < r.length:
                # valid hit in front of the target
                r.shorten(float(min_dists[i]))
            else:
                # no valid hit or behind/at target -> let ray reach the target sample point
                r.hit_distance = r.length

    def visible_fraction(self) -> float:
        """Return fraction of rays not blocked by any obstacle."""
        if self._hit_mask is None:
            return 1.0
        return float(np.sum(~self._hit_mask)) / len(self._hit_mask)

    def as_geometry(self):
        """Return arrays for visualization (origins, endpoints, colors)."""
        origins = np.array([r.origin for r in self.rays])
        ends = np.array([r.end_point() for r in self.rays])
        colors = np.array([[255, 0, 0] if hit else [0, 255, 0] for hit in self._hit_mask])
        return origins, ends, colors


# ============================================================
# --- OcclusionAnalyzer
# ============================================================

class OcclusionAnalyzer:
    """Computes occlusion only between a detection and its own source sensor."""

    def __init__(self, obstacles: List[Obstacle], sensors: Dict[int, "Source"]):
        self.obstacles = obstacles
        self.sensors = sensors

    def analyze_source_frame(
        self, source: "Source", joints: np.ndarray, n_samples: int = 50
    ) -> Dict[int, float]:
        """
        Compute per-joint visibility for a skeleton as seen from its own source.

        Args:
            source: Source sensor object (contains id, position, ypr)
            joints: (N,3) array of joint positions for that sensor’s detection
            n_samples: number of sampled rays per joint
        Returns:
            dict[int, float]: per-joint visibility fraction (0–1)
        """
        sensor_pos = source.position
        per_joint_visibility = {}

        for j_idx, joint_center in enumerate(joints):
            js = JointSphere(joint_center)
            js.sample_points(n_samples)
            dirs = js.directions_from(sensor_pos)

            bundle = RayBundle(sensor_id=source.sensor_id, frame_idx=0)
            for sample_point in js.samples:
                bundle.rays.append(Ray(sensor_pos, sample_point))


            bundle.cast_against(self.obstacles)
            per_joint_visibility[j_idx] = bundle.visible_fraction()

        return per_joint_visibility

    def analyze_all_sources(
        self, detections: Dict[int, np.ndarray], n_samples: int = 50
    ) -> Dict[int, Dict[int, float]]:
        """
        Compute per-source, per-joint visibility for all active sources in a frame.

        Args:
            detections: {sensor_id: joints (N,3)}
        Returns:
            {sensor_id: {joint_idx: visibility_fraction}}
        """
        results = {}
        for sid, joints in detections.items():
            if sid in self.sensors:
                results[sid] = self.analyze_source_frame(self.sensors[sid], joints, n_samples)
        return results

# ============================================================
# --- OcclusionVisualizer
# ============================================================

import rerun as rr

class OcclusionVisualizer:
    """Handles rerun-based visualization of rays, cuboids, and occlusion maps."""

    def __init__(self, app_name="Occlusion Debug Viewer"):
        rr.init(app_name, spawn=True)

    def draw_obstacles(self, obstacles: List[Obstacle], prefix="obstacles", color=(180, 180, 180)):
        """Draw all cuboid obstacles (environment + body) as Boxes3D."""
        from scipy.spatial.transform import Rotation
        for i, o in enumerate(obstacles):
            Rm = o.rotation.copy()
            det = np.linalg.det(Rm)

            if det < 0:
                Rm[:, 2] *= -1.0
            elif det == 0 or not np.isfinite(det):
                Rm = np.eye(3)

            quat = Rotation.from_matrix(Rm).as_quat()
            rr.log(
                f"{prefix}/{i}",
                rr.Boxes3D(
                    centers=[o.center],
                    half_sizes=[o.dims / 2.0],
                    quaternions=[quat],
                    colors=[color if o.category == "env" else (255, 200, 150)],
                ),
            )


    def draw_bundle(self, bundle: RayBundle, name=None):
        """Draw rays from a RayBundle with red = occluded, green = visible, and draw origins as points."""
        name = name or f"sensor_{bundle.sensor_id:02d}/rays_frame_{bundle.frame_idx}"
        origins, ends, colors = bundle.as_geometry()
        if len(origins) == 0:
            return
        rr.log(
            name,
            rr.LineStrips3D(
                np.stack([origins, ends], axis=1),
                colors=colors,
                radii=0.0006,
            ),
        )
        # Draw ray origins as small points for extra clarity
        rr.log(
            f"{name}_origins",
            rr.Points3D(origins, colors=[(255, 255, 0)], radii=0.003)
        )

    def draw_joints(self, joints: np.ndarray, radii=0.01, color=(0, 0, 255), name="joints"):
        """Visualize joint centers as points (default blue)."""
        rr.log(name, rr.Points3D(joints, colors=[color], radii=radii))

    def draw_occlusion_map(self, occlusion_map: np.ndarray):
        """
        Draw per-joint occlusion map as a simple heat color bar (frame vs joint).
        1 = fully visible (green), 0 = occluded (red)
        """
        if occlusion_map.size == 0:
            return
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        cmap = cm.get_cmap("RdYlGn")
        for fidx in range(occlusion_map.shape[0]):
            colors = (cmap(occlusion_map[fidx])[:, :3] * 255).astype(np.uint8)
            rr.log(
                f"occlusion_map/frame_{fidx}",
                rr.Points3D(
                    positions=np.stack(
                        [np.full_like(occlusion_map[fidx], fidx),  # x = frame
                         np.arange(occlusion_map.shape[1]),         # y = joint
                         np.zeros_like(occlusion_map[fidx])],       # z = 0
                        axis=1
                    ),
                    colors=colors,
                    radii=0.01,
                ),
            )

