import rerun as rr
import numpy as np
import h5py


class EnvironmentVisualizer:
    """
    Visualizes static environment objects (shelves, mirrors, counters, etc.)
    from the HDF5 /environment group, including meshes, outlines, and optional labels.
    """

    def __init__(self, show_environment=True, show_labels=False, show_outline=True, show_floor=False):
        self.show_environment = show_environment
        self.show_labels = show_labels
        self.show_outline = show_outline
        self.show_floor = show_floor

    # ----------------------------------------------------------
    # --- Color utility
    # ----------------------------------------------------------
    def _color_for_object(self, name: str):
        lname = name.lower()
        color_map = {
            "shelf": [255, 255, 255, 0.5],        # semi-transparent blue
            "mirror": [255, 255, 255, 1.0],  # silver/white
            "counter": [255, 255, 255, 1.0],      # semi-transparent green
            "entrance": [255, 255, 255, 1.0], # white
        }
        return color_map.get(lname, [255, 255, 255, 0.3])

    # ----------------------------------------------------------
    # --- Draw a single object
    # ----------------------------------------------------------
    def _visualize_single_object(self, key: str, obj_group):
        """Render one environment object, uniquely identified by its key."""
        name = obj_group.attrs.get("name", key)
        if "vertices" not in obj_group:
            return

        verts = np.array(obj_group["vertices"][:], dtype=np.float32)
        color = self._color_for_object(name)

        # Define cuboid faces (triangulated)
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [4, 7, 3, 0],
        ]
        triangles = np.array(
            [[f[0], f[1], f[2]] for f in faces] + [[f[0], f[2], f[3]] for f in faces],
            dtype=np.uint32,
        )

        base_path = f"environment/{key}"

        # Mesh
        rr.log(
            f"{base_path}/mesh",
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=triangles,
                vertex_colors=[[c / 255.0 for c in color[:3]]] * len(verts),
            ),
            static=True,
        )

        # Outline edges
        if self.show_outline:
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ]
            rr.log(
                f"{base_path}/edges",
                rr.LineStrips3D([[verts[a], verts[b]] for a, b in edges],
                                colors=[[0, 0, 0]] * len(edges)),
                static=True,
            )

        # Optional name label
        if self.show_labels:
            center = verts.mean(axis=0)
            rr.log(
                f"{base_path}/label",
                rr.Points3D([center], labels=[name], radii=0.03),
                static=True,
            )

    # ----------------------------------------------------------
    # --- Visualize all environment objects
    # ----------------------------------------------------------
    def visualize_all_objects(self, hdf: h5py.File):
        """Visualize all environment objects stored under /environment."""
        if not self.show_environment or "environment" not in hdf:
            print("[EnvironmentVisualizer] No environment found.")
            return

        env_group = hdf["environment"]
        print(f"[EnvironmentVisualizer] Rendering {len(env_group)} environment objects...")

        for key, obj_group in env_group.items():
            self._visualize_single_object(key, obj_group)

        # Optional floor plane
        if self.show_floor:
            self._draw_floor_plane()

    # ----------------------------------------------------------
    # --- Floor helper
    # ----------------------------------------------------------
    def _draw_floor_plane(self):
        """Draws a simple floor base to give spatial context."""
        ground = np.array([
            [0, 0, 0],
            [8, 0, 0],
            [8, 6, 0],
            [0, 6, 0],
        ])
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        rr.log(
            "environment/floor",
            rr.Mesh3D(
                vertex_positions=ground,
                triangle_indices=triangles,
                vertex_colors=[[240, 240, 240, 40]] * 4,
            ),
            static=True,
        )
        rr.log(
            "environment/floor/edges",
            rr.LineStrips3D([
                [ground[0], ground[1]],
                [ground[1], ground[2]],
                [ground[2], ground[3]],
                [ground[3], ground[0]],
            ], colors=[[50, 50, 50]] * 4),
            static=True,
        )
