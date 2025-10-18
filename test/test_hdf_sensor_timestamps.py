import h5py

path = r"C:\Users\zulkafil\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5"

with h5py.File(path, "r") as f:
    for tid, g in f["poses_raw"].items():
        for sid in g:
            src_group = g[sid]
            if str(tid) not in src_group:
                print("Missing subtrack:", tid, sid)
                continue
            sub = src_group[str(tid)]
            if "timestamps" not in sub:
                print("Missing timestamps:", tid, sid)
