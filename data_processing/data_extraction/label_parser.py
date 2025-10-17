import h5py
import re

class Label:
    """Represents a single TMV label with 3 parts: location, arm action, and interaction."""
    def __init__(self, raw):
        self.raw = raw
        self.location = None
        self.arm_action = None
        self.interaction = None
        self._parse()

    def _parse(self):
        """
        Parse TMV-style raw label strings of the form:
        "Customer Movement & Locations: Standing at the Shelf, Customer Arm Actions: Pick or Place a Hat, Interaction With Shopkeeper: Not Interacting"
        Extracts each field explicitly.
        """
        text = self.raw

        # Extract using regex patterns for each known prefix
        loc_match = re.search(r"Customer Movement & Locations:\s*([^,|]+)", text)
        arm_match = re.search(r"Customer Arm Actions:\s*([^,|]+)", text)
        inter_match = re.search(r"Interaction With Shopkeeper:\s*([^,|]+)", text)

        self.location = loc_match.group(1).strip() if loc_match else "None"
        self.arm_action = arm_match.group(1).strip() if arm_match else "None"
        self.interaction = inter_match.group(1).strip() if inter_match else "None"


    def as_dict(self):
        return {
            "raw": self.raw,
            "location": self.location,
            "arm_action": self.arm_action,
            "interaction": self.interaction
        }

    def __repr__(self):
        return f"<Label loc={self.location}, arm={self.arm_action}, inter={self.interaction}>"

# -----------------------------------------------------

def list_label_ids(hdf: h5py.File):
    """Return all tracking IDs that have labels."""
    return list(map(int, hdf["labels"].keys())) if "labels" in hdf else []

def read_labels_for_id(hdf: h5py.File, tid: int):
    """Read and parse labels for a tracking ID as Label objects."""
    path = f"labels/{tid}"
    if path not in hdf:
        return []
    grp = hdf[path]
    starts = grp["start_times"][:]
    durations = grp["durations"][:]
    raw_actions = [a.decode() if isinstance(a, bytes) else a for a in grp["actions"][:]]
    labels = [Label(a) for a in raw_actions]
    return list(zip(starts, durations, labels))
