import json
import numpy as np
import h5py
from pathlib import Path
from data_processing.data_extraction.hdf5_utils import ensure_group, create_or_replace_dataset


def insert_environment(hdf: h5py.File, env_json_path: str):
    """
    Read environment objects from JSON and insert into /environment group.
    Each object has vertices[][3] and a 'name' attribute.

    JSON format example:
    {
        "objects": [
            {
                "name": "shelf_1",
                "vertices": [[x, y, z], ...]
            },
            {
                "name": "mirror_1",
                "vertices": [[x, y, z], ...]
            }
        ]
    }
    """
    env_json_path = Path(env_json_path)
    if not env_json_path.exists():
        print(f"[EnvironmentExtractor] Warning: file not found at {env_json_path}")
        return

    with open(env_json_path, "r") as f:
        data = json.load(f)

    objects = data.get("objects", [])
    if not objects:
        print(f"[EnvironmentExtractor] No objects found in {env_json_path}")
        return

    env_group = ensure_group(hdf, "environment")

    for i, obj in enumerate(objects):
        name = obj.get("name", f"object_{i}")
        verts = np.array(obj.get("vertices", []), dtype="f4")

        obj_group = ensure_group(env_group, f"{name}_{i}")
        create_or_replace_dataset(obj_group, "vertices", verts)
        obj_group.attrs["name"] = name

    print(f"[EnvironmentExtractor] Inserted {len(objects)} environment objects from {env_json_path}")
