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

# Running Nested Scripts (FOR NOW)
Use `python -m <dir.dir.script>` in the root of the repository.

Example:
```bash
(rerun_env) PS C:\...\individual_customers_codebase> 
python -m test.test_configs
python -m data_processing.data_extraction.configs_generator
```

```(rerun_env) PS C:\...\Desktop\individual_customers_codebase> python -m  data_processing.data_extraction.basic_extractor --config C:\...\Desktop\individual_customers_codebase\configs\2025-03-16_s2_merged_tracked_config.yaml```

```(rerun_env) PS C:\...\individual_customers_codebase> python -m data_processing.data_visualization.skeleton_viewer --file C:\...\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5 --mode poses_fused --show-joints --show-bones```

```(rerun_env) PS C:\...\individual_customers_codebase> python -m data_processing.data_visualization.skeleton_viewer --file C:\...\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5 --mode poses_fused --show-joints --show-bones --show-labels```

```(rerun_env) PS C:\...\individual_customers_codebase> python -m data_processing.data_visualization.skeleton_viewer --file C:\...\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5 --mode poses_fused --show-joints --show-bones --show-labels --show-environment```

```(rerun_env) PS C:\...\individual_customers_codebase> python -m data_processing.data_visualization.skeleton_viewer --file C:\...\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5 --mode poses_fused --show-joints --show-bones --show-labels --show-environment --show_sensors```

```(rerun_env) PS C:\...\individual_customers_codebase> python -m data_processing.data_visualization.skeleton_viewer --file C:\...\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5 --mode poses_fused --show-joints --show-bones --show-labels --show-environment --show_sensors --active-sensors 1 2 3 4 5 6 --selected-sensors 3```