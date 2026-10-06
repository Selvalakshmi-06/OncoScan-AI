import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

from src.explainability.gradcam import generate_gradcam_for_image
from src.preprocessing.data_loader import CLASS_NAMES

# ============================================================
# OncoScan AI - Flask Backend
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "oncoscan_efficientnetb0.keras"
)
IMG_SIZE = (224, 224)

UPLOAD_DIR = "backend/uploads"
GRADCAM_DIR = "backend/gradcam_results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

# ============================================================
# Load Trained Model
# ============================================================

print("\nLoading OncoScan AI model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Model path:", MODEL_PATH)
print("Classes:", CLASS_NAMES)


# ============================================================
# Home Route
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "message": "OncoScan AI backend is running!",
        "status": "success"
    })


# ============================================================
# Health Check
# ============================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


# ============================================================
# Prediction + Grad-CAM Route
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check uploaded image
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    try:
        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        filename = os.path.basename(file.filename)
        image_path = os.path.join(
            UPLOAD_DIR,
            filename
        )

        file.save(image_path)

        # ----------------------------------------------------
        # Prepare image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            IMG_SIZE
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # Generate prediction + Grad-CAM
        # ----------------------------------------------------

        (
            heatmap,
            predicted_class,
            probabilities
        ) = generate_gradcam_for_image(
            model,
            image_array
        )

        predicted_name = CLASS_NAMES[
            predicted_class
        ]

        confidence = (
            float(probabilities[predicted_class])
            * 100
        )

        # ----------------------------------------------------
        # Probability results
        # ----------------------------------------------------

        class_probabilities = {}

        for index, class_name in enumerate(
            CLASS_NAMES
        ):
            class_probabilities[class_name] = round(
                float(probabilities[index]) * 100,
                2
            )

        # ----------------------------------------------------
        # Create Grad-CAM image
        # ----------------------------------------------------

        original_array = np.array(
            image
        )

        gradcam_filename = (
            "gradcam_result.png"
        )

        gradcam_path = os.path.join(
            GRADCAM_DIR,
            gradcam_filename
        )

        plt_figure = tf.keras.utils.array_to_img(
            original_array
        )

        # Use matplotlib for heatmap overlay
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))

        plt.imshow(
            original_array
        )

        plt.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45
        )

        plt.title(
            f"Prediction: {predicted_name}\n"
            f"Confidence: {confidence:.2f}%"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            gradcam_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({
            "prediction": predicted_name,
            "confidence": round(
                confidence,
                2
            ),
            "probabilities": class_probabilities,
            "gradcam_url": "/gradcam/gradcam_result.png"
        })

    except Exception as error:

        print(
            "\nPrediction error:",
            error
        )

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# Grad-CAM Image Route
# ============================================================


@app.route("/gradcam/<filename>")
def serve_gradcam(filename):
    file_path = os.path.join(GRADCAM_DIR, filename)

    print("Grad-CAM requested:", filename)
    print("Grad-CAM path:", file_path)
    print("File exists:", os.path.exists(file_path))

    return send_from_directory(
        os.path.abspath(GRADCAM_DIR),
        filename
    )
# ============================================================
# Run Flask Server
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )