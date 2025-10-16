import json

with open("common_utils/hatshop_individual_customers_index.json") as f:
    data = json.load(f)

wrong_tmv_file_extensions = [
    e for e in data["entries"]
    if not e.get("Tmv File", "").endswith(".tmv")
]

wrong_bag_file_extensions = [
    e for e in data["entries"]
    if not e.get("Bag File", "").endswith(".bag")
]

print(f"{len(wrong_bag_file_extensions)} entries with wrong bag file extension:")
for w in wrong_bag_file_extensions:
    print(w["Index"], w["Bag File"])

print(f"{len(wrong_tmv_file_extensions)} entries with wrong tmv file extension:")
for w in wrong_tmv_file_extensions:
    print(w["Index"], w["Tmv File"])

