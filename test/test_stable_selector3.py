import rerun as rr
import h5py
from data_processing.data_extraction.stable_selector2 import StableSelector

# path = r"C:\Users\zulkafil\Desktop\individual_customers_codebase\dataset\2025-03-16_s2_merged_tracked.hdf5"
path = r"C:\Users\zulkafil\Desktop\individual_customers_codebase\dataset\2025-03-16_s4_merged_tracked.hdf5"


with h5py.File(path, "r") as f_tmp:
    tid = list(f_tmp["poses_raw"].keys())[0]

selector = StableSelector()
with h5py.File(path, "r") as f:
    selector.select_and_visualize(f, tid, rr)
