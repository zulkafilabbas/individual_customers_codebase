import h5py

def show_structure(g, indent=0):
    """Recursively print the structure of the hdf5 group/dataset tree."""
    for k in g.keys():
        v = g[k]
        if isinstance(v, h5py.Group):
            print("  " * indent + f"{k}/ (Group)")
            show_structure(v, indent + 1)
        else:
            print("  " * indent + f"{k} (Dataset) shape={v.shape}, dtype={v.dtype}")

# Open the file for both structure and detailed access
file_path = r"C:\Users\zulkafil\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5"
with h5py.File(file_path, "r") as f:
    print("---- HDF File Structure ----")
    show_structure(f)
    print("\n---- Sensor Structure ----")
    show_structure(f["sensors"])

    print("\n---- Example pose_raw breakdown ----")
    tid = list(f["poses_raw"].keys())[0]
    print("TID:", tid)
    print("pose_raw/TID available sources:", list(f[f"poses_raw/{tid}"].keys()))
    for sid in f[f"poses_raw/{tid}"].keys():
        print(f"Source: {sid} | Groups:", list(f[f'poses_raw/{tid}/{sid}'].keys()))
