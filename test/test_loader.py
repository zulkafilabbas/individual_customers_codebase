from common_utils.loader import JsonLoader

loader = JsonLoader("common_utils")
env = loader.get_environment()
extrinsics = loader.get_extrinsics("2025")
skeleton = loader.get_skeleton()

print(loader.get_extrinsics())
print(loader.get_environment())
print(loader.get_skeleton()["joints"])
print(loader.get_skeleton()["bones"])

extractor_settings = loader.get_data_extraction_settings()
print(extractor_settings)
