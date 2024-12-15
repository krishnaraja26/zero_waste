from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import gdown
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ANNOTATED_FOLDER"] = "static/annotated"
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}

# Ensure static folders exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["ANNOTATED_FOLDER"], exist_ok=True)

# Download weights from Google Drive
weights_url = "https://drive.google.com/uc?id=11HUMKP9qh9nvpsHHJ6BlVueSfESimfWQ"  # Public file link converted
weights_path = "best.pt"

# Ensure weights are downloaded
if not os.path.exists(weights_path):
    print("Downloading model weights...")
    gdown.download(weights_url, weights_path, quiet=False)

# Load the YOLO model with your custom-trained weights
model = YOLO(weights_path)

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def draw_boxes(image_path, predictions):
    """
    Draw bounding boxes on the image for each class.
    Returns the path to the annotated image and a list of annotations for that class.
    """
    image = cv2.imread(image_path)
    annotations = []  # To store coordinates and labels for the current class

    for pred in predictions:
        x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        confidence = pred["confidence"]
        label = f"{pred['class_name']} {confidence * 100:.2f}%"

        # Draw bounding box on image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Store bounding box and label for later use in HTML/JS
        annotations.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, 
            "label": label,
            "class_name": pred["class_name"]
        })

    # Save the annotated image
    annotated_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_annotated_{pred['class_name']}.jpg"
    annotated_image_path = os.path.join(app.config["ANNOTATED_FOLDER"], annotated_filename)
    cv2.imwrite(annotated_image_path, image)

    return annotated_image_path, annotations

@app.route("/")
def index():
    """Render the main page for image upload."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image uploads and run predictions."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Predict with YOLO
        results = model(filepath)

        # Parse predictions
        predictions_data = []
        class_to_predictions = {}
        for box in results[0].boxes:
            xyxy = box.xyxy.squeeze().cpu().numpy()  # Convert tensor to numpy array
            conf = box.conf.squeeze().item()
            cls = int(box.cls.squeeze().item())
            class_name = model.names[cls]

            prediction = {
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3]),
                "confidence": float(conf),
                "class_id": cls,
                "class_name": class_name,
            }
            predictions_data.append(prediction)

            # Group predictions by class name
            if class_name not in class_to_predictions:
                class_to_predictions[class_name] = []
            class_to_predictions[class_name].append(prediction)

        # Create separate annotated images for each class
        annotated_image_paths = []
        all_annotations = []
        for class_name, preds in class_to_predictions.items():
            annotated_image_path, annotations = draw_boxes(filepath, preds)
            annotated_image_paths.append({
                "class_name": class_name,
                "image_path": f"/{annotated_image_path}",  # Correct path for static files
                "confidence": preds[0]["confidence"],  # Assuming confidence for this class is the same for all
                "annotations": annotations
            })
        return jsonify({
            "image_path": f"/{app.config['UPLOAD_FOLDER']}/{filename}",
            "annotations": all_annotations,
            "annotated_images": annotated_image_paths
        })
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    app.run(debug=True)
