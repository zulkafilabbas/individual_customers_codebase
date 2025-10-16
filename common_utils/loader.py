import json
from pathlib import Path

class JsonLoader:
    def __init__(self, base_dir):
        self.base = Path(base_dir)
        self._cache = {}

    def _load_json(self, name):
        if name not in self._cache:
            path = self.base / name
            with open(path, "r", encoding="utf-8") as f:
                self._cache[name] = json.load(f)
        return self._cache[name]

    def get_environment(self):
        return self._load_json("hatshop_environment_objects.json")

    def get_customers(self):
        return self._load_json("hatshop_individual_customers_index.json")

    def get_extrinsics(self, year="2025"):
        data = self._load_json("hatshop_extrinsic_calibration.json")
        return data.get(str(year), {})

    def get_skeleton(self):
        data = self._load_json("azure_kinect_skeleton.json")

        # main hierarchy: each joints name and its parent index (defines the tree)
        joints = {}
        for k, v in data["joints"].items():
            joints[int(k)] = tuple(v)
        
        # list of (child, parent) pairs, handy for downstream tasks like visualization
        bones = []
        for jid, (_, parent) in joints.items():
            if parent is not None:
                bones.append((jid, parent))

        # inverse hierarchy: for each parent, a list of its child joints (easier for top-down traversal)
        children = {}
        for jid, (_, parent) in joints.items():
            if parent is not None:
                children.setdefault(parent, []).append(jid)

        return {"joints": joints, "bones": bones, "children": children}
