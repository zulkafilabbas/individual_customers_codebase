import h5py
f = h5py.File(r"C:\Users\zulkafil\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5", "r")

def show_structure(g, indent=0):
    for k, v in g.items():
        print("  " * indent + f"{k}/ ({'Group' if isinstance(v, h5py.Group) else 'Dataset'})")
        if isinstance(v, h5py.Group):
            show_structure(v, indent + 1)

show_structure(f["sensors"])
f.close()
