import h5py
import numpy as np

def require_group(h5file, path):
    """Create nested groups if needed and return the final group."""
    group = h5file
    for part in path.strip("/").split("/"):
        group = group.require_group(part)
    return group

def ensure_group(h5file, path):
    """Create nested groups if needed and return the final group."""
    group = h5file
    for part in path.strip("/").split("/"):
        group = group.require_group(part)
    return group

def create_or_replace_dataset(group, name, data, compression="gzip"):
    """Create or replace a dataset safely."""
    if name in group:
        del group[name]
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    group.create_dataset(name, data=data, compression=compression)

def create_expandable_dataset(group, name, shape, dtype):
    """Create an expandable dataset if it doesn't exist."""
    if name not in group:
        group.create_dataset(name, shape, maxshape=(None,) + shape[1:], dtype=dtype)

def write_attrs(obj, attrs_dict):
    """Write multiple attributes to a group or dataset."""
    for key, val in attrs_dict.items():
        obj.attrs[key] = val


def exists(group, name):
    """Check if a dataset or subgroup exists."""
    return name in group

# Plan for hdf5_utils.py
# 1. Safe nested group creation
#    Needed for `/poses_raw/tid/source_xx/track_id`, `/poses_fused/tid`, `/poses_stable/tid`, `/environment/object_name`, etc.
#    require_group(h5file, path) ensures all intermediate groups exist.

# 2. Dataset creation / overwrite control
#    Used for datasets like `timestamps`, `joints`, `detection_indices`, `track_states`, `vertices`, `xyz`, `ypr`, etc.
#    create_or_replace_dataset(group, name, data, compression="gzip")

# 3. Attribute writer
#    Used for metadata (e.g., `bag_start_time`, `label_anchor`), environment (`name`), and other static attributes.
#    write_attrs(obj, attrs_dict)

# 4. Existence check
#    Sometimes needed to skip re-writing already created groups or datasets.
#    exists(group, name)