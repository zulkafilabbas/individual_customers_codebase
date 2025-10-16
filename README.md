# HDF5 Structure

```bash
<file>.hdf5
├── metadata
│   ├── attrs:
│   │   ├── bag_start_time
│   │   ├── video_start_time
│   │   ├── label_anchor = "bag_start_time"
│   │   └── (optional) tmv_file
│   ├── joint_names[32]
│   └── extracted_ids[n]
│
├── poses_raw
│   └── <tracking_id>
│       └── source_<sid>
│           └── <tracking_id>
│               ├── timestamps[]
│               ├── joints[][32,3]
│               ├── detection_indices[]
│               └── track_states[][32]
│
├── poses_fused
│   └── <tracking_id>
│       ├── timestamps[]
│       ├── joints[][32,3]
│       ├── detection_indices[]
│       └── track_states[][32]
│
├── poses_stable
│   └── <tracking_id>
│       ├── timestamps[]
│       ├── joints[][32,3]
│       ├── detection_indices[]
│       └── track_states[][32]
│
├── images
│   └── *<image_topic>*
│       └── timestamps[]
│
├── labels (optional)
│   └── <tracking_id>
│       ├── start_times[]
│       ├── durations[]
│       └── actions[]
│
├── environment (optional)
│   └── <object_name_i>
│       ├── vertices[][3]
│       └── attr name = "object_name"
│
└── sensors
    ├── 2024
    │   └── sensor_windows_xx
    │       ├── xyz[3]
    │       └── ypr[3]
    └── 2025
        └── sensor_windows_xx
            ├── xyz[3]
            └── ypr[3]
```