import os
import io
import base64
import random
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Find best weights or fallback
WEIGHT_CANDIDATES = [
    BASE_DIR / "runs" / "detect" / "runs" / "urban_object_detection_1500" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "urban_object_detection_1500" / "weights" / "best.pt",
    BASE_DIR / "yolov8n.pt"
]

MODEL_PATH = None
for path in WEIGHT_CANDIDATES:
    if path.exists():
        MODEL_PATH = str(path)
        break

print(f"[*] Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Class names (BDD100K urban object classes)
CLASS_NAMES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]

# Color palette for classes (RGB)
CLASS_COLORS = {
    "person": (255, 59, 48),       # Red
    "rider": (255, 149, 0),       # Orange
    "car": (52, 199, 89),         # Green
    "bus": (0, 199, 190),         # Teal
    "truck": (48, 176, 255),      # Light Blue
    "bike": (88, 86, 214),        # Purple
    "motor": (175, 82, 222),      # Pinkish Purple
    "traffic light": (255, 204, 0),# Yellow
    "traffic sign": (255, 45, 85), # Pink/Rose
    "train": (162, 132, 94)       # Brown
}

def get_color(class_name):
    return CLASS_COLORS.get(class_name.lower(), (0, 230, 255))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files and "sample_path" not in request.form:
            return jsonify({"error": "No image file provided"}), 400

        conf_threshold = float(request.form.get("confidence", 0.25))

        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif "sample_path" in request.form:
            sample_rel = request.form.get("sample_path")
            sample_full = BASE_DIR / sample_rel
            if not sample_full.exists():
                return jsonify({"error": f"Sample image not found: {sample_rel}"}), 404
            img = cv2.imread(str(sample_full))
        else:
            return jsonify({"error": "Invalid request parameters"}), 400

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Original image dimensions
        orig_h, orig_w = img.shape[:2]

        # Run inference
        results = model.predict(source=img, conf=conf_threshold, verbose=False)
        res = results[0]

        annotated_img = img.copy()
        detections = []
        class_counts = {}

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in coords]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                if hasattr(res, "names") and cls_id in res.names:
                    cls_name = res.names[cls_id]
                elif cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                else:
                    cls_name = f"class_{cls_id}"

                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                color = get_color(cls_name)
                # BGR for OpenCV
                bgr_color = (color[2], color[1], color[0])

                # Draw bounding box
                thickness = max(2, int(min(orig_h, orig_w) / 300))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), bgr_color, thickness)

                # Draw label box
                label = f"{cls_name} {conf:.2f}"
                font_scale = max(0.4, min(orig_h, orig_w) / 1000)
                font_thick = max(1, int(thickness / 2))
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

                # Label background rectangle
                lbl_y1 = max(0, y1 - text_h - 8)
                lbl_y2 = y1
                cv2.rectangle(annotated_img, (x1, lbl_y1), (x1 + text_w + 10, lbl_y2), bgr_color, -1)
                
                # Text in white/dark depending on color brightness
                text_color = (255, 255, 255)
                cv2.putText(annotated_img, label, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "box": [x1, y1, x2, y2]
                })

        # Encode annotated image to JPEG base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "total_detections": len(detections),
            "class_counts": class_counts,
            "detections": detections,
            "dimensions": {"width": orig_w, "height": orig_h},
            "model_used": os.path.basename(MODEL_PATH)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/samples", methods=["GET"])
def get_samples():
    val_dir = BASE_DIR / "dataset" / "val" / "images"
    samples = []
    if val_dir.exists():
        images = list(val_dir.glob("*.jpg"))
        if images:
            # Pick 6 random sample images
            selected = random.sample(images, min(6, len(images)))
            for img_p in selected:
                rel_path = f"dataset/val/images/{img_p.name}"
                samples.append({
                    "name": img_p.name,
                    "path": rel_path
                })
    return jsonify({"samples": samples})

@app.route("/dataset/val/images/<filename>")
def serve_sample_img(filename):
    val_dir = BASE_DIR / "dataset" / "val" / "images"
    return send_from_directory(val_dir, filename)

if __name__ == "__main__":
    print("Starting Flask Web UI for Urban Object Detection...")
    app.run(host="127.0.0.1", port=5000, debug=True)
