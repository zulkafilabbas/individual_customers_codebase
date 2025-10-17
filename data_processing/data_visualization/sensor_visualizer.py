import rerun as rr
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R


class SensorVisualizer:
    """Visualizes camera sensors (xyz + ypr) from /sensors/<year>/<sensor_name>/."""

    def __init__(self, show_sensors=True, axis_length=0.1, show_frustum=True, box_size=0.15):
        self.show_sensors = show_sensors
        self.axis_length = axis_length
        self.show_frustum = show_frustum
        self.box_size = np.array([box_size, box_size, box_size])
        self.state_colors = {
            "inactive": [100, 100, 100, 255],
            "active": [255, 255, 0, 255],
            "selected": [0, 255, 0, 255],
        }
        self.hdf = None # 

    def _draw_sensor_axes(self, path, position, rotation):
        """Draw coordinate axes for one sensor aligned with frustum (Z+ forward)."""

        # Flip Z axis to align with Azure depth coordinate system
        # adjusted_rot = rotation.copy()
        # adjusted_rot[:, 2] *= -1  # invert forward direction

        # Rotate 180° around Y to match Rerun camera frustum orientation
        # align_rot = R.from_euler("y", np.pi, degrees=False).as_matrix()
        # adjusted_rot = rotation @ align_rot

        # Combine: 180° Y (flip forward) + 90° X (tilt upward)
        # align_rot = R.from_euler("XY", [np.pi / 2, np.pi], degrees=False).as_matrix()
        # adjusted_rot = rotation @ align_rot 

        # rr.log(
        #     f"{path}/axes/local",
        #     rr.Arrows3D(
        #         origins=np.zeros((3, 3)),
        #         # vectors=rotation * self.axis_length,
        #         vectors=adjusted_rot * self.axis_length,
        #         colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        #     ),
        # )

        # Above commented out part was just annoying to adjust
        # A simpler approach is just to point the axes in the direction of the frustum
        # The frustum is already pointed correctly, so use that to know where the calibrated depth sensor is pointing
        # Note this is not RGB, this is Depth!
        # See: https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/coordinate-systems#depth-and-color-camera

        rr.log(
            f"{path}/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),  # local origin of the sensor
                vectors=np.eye(3) * self.axis_length,  # identity = local X/Y/Z
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def _draw_frustum_box(self, path, position, ypr, state="inactive"):
        color = self.state_colors.get(state, self.state_colors["inactive"])
        rot = R.from_euler("ZYX", ypr, degrees=False)
        quat = rr.Quaternion(xyzw=rot.as_quat())

        rr.log(
            path,
            rr.Transform3D(translation=position.tolist(), rotation=quat),
        )

        # Ensure alpha < 255 for transparency
        color = np.array(color, dtype=np.uint8)
        color[3] = 120  # ≈ 50% transparent (tune 0–255)

        rr.log(
            f"{path}/box",
            rr.Boxes3D(
                centers=[[0, 0, 0]],
                half_sizes=[self.box_size / 2],
                colors=[color.tolist()],
                fill_mode="solid",
            ),
        )

        if self.show_frustum:
            rr.log(
                f"{path}/camera",
                rr.Pinhole(
                    resolution=[640, 576],
                    focal_length=500,
                    principal_point=[320, 288],
                    image_plane_distance=0.1,
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )

    def visualize_sensors(self, hdf, active_sensors=None, selected_sensors=None):
        active_sensors = active_sensors or set()
        selected_sensors = selected_sensors or set()

        if not self.show_sensors or "sensors" not in hdf:
            print("[SensorVisualizer] No sensors found in HDF5.")
            return

        rr.set_time_seconds("frame", 0.0)  # Ensure sensors appear on the same timeline as skeletons

        sensors_group = hdf["sensors"]

        for year_key, year_group in sensors_group.items():
            if not isinstance(year_group, h5py.Group):
                continue

            print(f"[SensorVisualizer] Rendering sensors for year {year_key}...")
            for sensor_name, sensor_group in year_group.items():
                if "xyz" in sensor_group and "ypr" in sensor_group:
                    pos = sensor_group["xyz"][:]
                    ypr = sensor_group["ypr"][:]  # already in radians
                    rot = R.from_euler("zyx", ypr, degrees=False).as_matrix()
                    # print(f"{sensor_name} pos={pos}, ypr={ypr}")

                    rr.log("sensors/world_positions",
                        rr.Points3D([pos], colors=[[255, 150, 0]], radii=[0.008]))

                    rr.log(
                        f"sensors/{year_key}/{sensor_name}/label",
                        rr.TextLog(f"{sensor_name} ({year_key})"),
                    )

                    # New: Determine state and draw colored frustum box
                    try:
                        sid = int(sensor_name.split("_")[-1])
                    except Exception:
                        sid = sensor_name
                    if sid in selected_sensors:
                        state = "selected"
                    elif sid in active_sensors:
                        state = "active"
                    else:
                        state = "inactive"

                    sensor_path = f"sensors/{year_key}/{sensor_name}"
                    self._draw_frustum_box(sensor_path, pos, ypr, state)
                    self._draw_sensor_axes(sensor_path, position=np.zeros(3), rotation=rot)


        print("[SensorVisualizer] All sensors rendered successfully.")

    def update_sensor_states(self, active_sensors=None, selected_sensors=None):
        """Optional per-frame update of sensor colors/states."""
        if self.hdf is None or "sensors" not in self.hdf:
            return

        active_sensors = active_sensors or set()
        selected_sensors = selected_sensors or set()

        for year_key, year_group in self.hdf["sensors"].items():
            for sensor_name, sensor_group in year_group.items():
                try:
                    sid = int(sensor_name.split("_")[-1])
                except Exception:
                    sid = sensor_name

                if sid in selected_sensors:
                    state = "selected"
                elif sid in active_sensors:
                    state = "active"
                else:
                    state = "inactive"

                pos = sensor_group["xyz"][:]
                ypr = sensor_group["ypr"][:]
                self._draw_frustum_box(f"sensors/{year_key}/{sensor_name}", pos, ypr, state)
