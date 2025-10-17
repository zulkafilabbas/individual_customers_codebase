import h5py
import numpy as np
from common_utils.loader import JsonLoader


def insert_sensors(hdf: h5py.File):
    """
    Insert all sensor extrinsics (xyz + ypr) into /sensors of the HDF5 file.
    Uses hatshop_extrinsic_calibration.json via JsonLoader.
    """
    loader = JsonLoader("common_utils")
    all_extrinsics = loader._load_json("hatshop_extrinsic_calibration.json")

    sensors_group = hdf.require_group("sensors")

    for year, sensors in all_extrinsics.items():
        year_group = sensors_group.require_group(str(year))
        for name, pose in sensors.items():
            if name == "map_to_world":
                continue
            g = year_group.require_group(name)
            # Overwrite existing datasets if they exist
            if "xyz" in g:
                del g["xyz"]
            if "ypr" in g:
                del g["ypr"]
            g.create_dataset("xyz", data=np.array(pose["xyz"], dtype="f4"))
            g.create_dataset("ypr", data=np.array(pose["ypr"], dtype="f4"))
            g.attrs["name"] = name

    print(f"[INFO] Inserted sensors for {len(all_extrinsics)} year(s).")


# Usage in basic_extractor.py:
# after writing metadata, labels, and environment
# ------------------------------------------------
# from data_processing.data_extraction.sensor_extractor import insert_sensors
# insert_sensors(hdf)
# ------------------------------------------------
