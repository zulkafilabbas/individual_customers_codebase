# data_processing/data_visualization/sensor_visualizer.py

import rerun as rr
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R


class SensorVisualizer:
    """Visualizes camera sensors (xyz + ypr) from /sensors/<year>/<sensor_name>/."""

    def __init__(self, show_sensors=True, axis_length=0.1):
        self.show_sensors = show_sensors
        self.axis_length = axis_length

    def _draw_sensor_axes(self, path, position, rotation):
        """Draw coordinate axes for one sensor."""
        rr.log(
            f"{path}/axes",
            rr.Arrows3D(
                origins=np.repeat([position], 3, axis=0),
                vectors=rotation * self.axis_length,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                # labels=["X", "Y", "Z"],
            ),
        )

    def visualize_sensors(self, hdf):
        """Recursively visualize all sensors under /sensors."""
        if not self.show_sensors or "sensors" not in hdf:
            print("[SensorVisualizer] No sensors found in HDF5.")
            return

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
                    print(f"{sensor_name} pos={pos}, ypr={ypr}")

                    rr.log(
                        f"sensors/{year_key}/{sensor_name}/origin",
                        rr.Points3D([pos], colors=[[255, 200, 0]], radii=[0.01]),
                    )
                    rr.log(
                        f"sensors/{year_key}/{sensor_name}/label",
                        rr.TextLog(f"{sensor_name} ({year_key})"),
                    )
                    self._draw_sensor_axes(f"sensors/{year_key}/{sensor_name}", pos, rot)

        print("[SensorVisualizer] All sensors rendered successfully.")
