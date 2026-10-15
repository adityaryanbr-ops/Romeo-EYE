import kagglehub, pathlib, shutil

# Choose the dataset you want
DATASET = "diprajpatra/uav-detection-images"

# Download dataset
path = kagglehub.dataset_download(DATASET)
print("Downloaded dataset to:", path)

# Project dataset folder
proj = pathlib.Path(r"C:\Users\adity\Downloads\drone-detection-starter")
dst  = proj / "datasets" / "drone_extra"

if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(path, dst)

print("✅ Copied dataset into:", dst)
