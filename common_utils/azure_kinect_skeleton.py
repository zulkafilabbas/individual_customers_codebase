# main hierarchy: each joints name and its parent index (defines the tree)
JOINTS = {
    0:  ("PELVIS", None),
    1:  ("SPINE_NAVEL", 0),
    2:  ("SPINE_CHEST", 1),
    3:  ("NECK", 2),
    4:  ("CLAVICLE_LEFT", 2),
    5:  ("SHOULDER_LEFT", 4),
    6:  ("ELBOW_LEFT", 5),
    7:  ("WRIST_LEFT", 6),
    8:  ("HAND_LEFT", 7),
    9:  ("HANDTIP_LEFT", 8),
    10: ("THUMB_LEFT", 7),
    11: ("CLAVICLE_RIGHT", 2),
    12: ("SHOULDER_RIGHT", 11),
    13: ("ELBOW_RIGHT", 12),
    14: ("WRIST_RIGHT", 13),
    15: ("HAND_RIGHT", 14),
    16: ("HANDTIP_RIGHT", 15),
    17: ("THUMB_RIGHT", 14),
    18: ("HIP_LEFT", 0),
    19: ("KNEE_LEFT", 18),
    20: ("ANKLE_LEFT", 19),
    21: ("FOOT_LEFT", 20),
    22: ("HIP_RIGHT", 0),
    23: ("KNEE_RIGHT", 22),
    24: ("ANKLE_RIGHT", 23),
    25: ("FOOT_RIGHT", 24),
    26: ("HEAD", 3),
    27: ("NOSE", 26),
    28: ("EYE_LEFT", 26),
    29: ("EAR_LEFT", 26),
    30: ("EYE_RIGHT", 26),
    31: ("EAR_RIGHT", 26),
}

# reverse hierarchy: for each parent, a list of its child joints (easier for top-down traversal)
CHILDREN = {}
for jid, (_, parent) in JOINTS.items():
    if parent is not None:
        CHILDREN.setdefault(parent, []).append(jid)

# list of (child, parent) pairs, handy for downstream tasks like visualization
BONES = [(i, parent) for i, (_, parent) in JOINTS.items() if parent is not None]

if __name__ == "__main__":
    print(JOINTS)
    print(CHILDREN)
    print(BONES)
