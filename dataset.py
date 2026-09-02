import kagglehub

# Download latest version
path = kagglehub.dataset_download("a7madmostafa/bdd100k-yolo")

print("Path to dataset files:", path)