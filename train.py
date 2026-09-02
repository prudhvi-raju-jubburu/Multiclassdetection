from ultralytics import YOLO
import os

print("=" * 70)
print("       MULTI-CLASS URBAN OBJECT DETECTION USING YOLO")
print("=" * 70)

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

print("\n" + "=" * 70)
print("STARTING MODEL TRAINING")
print("=" * 70)

results = model.train(
    data="data.yaml",
    epochs=8,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    patience=3,
    project="runs",
    name="urban_object_detection_1500",
    exist_ok=True,
    seed=42
)

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

best_model_path = os.path.join(
    "runs",
    "detect",
    "runs",
    "urban_object_detection_1500",
    "weights",
    "best.pt"
)

print("Best model saved at:")
print(best_model_path)

print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

# Load the best model only if you want to perform additional validation
best_model = YOLO(best_model_path)

metrics = best_model.val(
    data="data.yaml",
    split="val",
    imgsz=320,
    batch=4,
    device="cpu"
)

print(f"\nmAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)