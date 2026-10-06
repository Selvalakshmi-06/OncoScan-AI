import os
import sys
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, PROJECT_ROOT)

from src.explainability.gradcam import generate_gradcam_for_image
from src.preprocessing.data_loader import CLASS_NAMES


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "oncoscan_efficientnetb0.keras"
)

IMG_SIZE = (224, 224)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="OncoScan AI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_oncoscan_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# Header
# ============================================================

st.title("🧠 OncoScan AI")

st.subheader(
    "AI-Powered Tumor Detection with Explainable AI"
)

st.write(
    "Upload a Brain MRI image to detect the tumor class "
    "and generate a Grad-CAM explanation."
)


# ============================================================
# Model Loading
# ============================================================

with st.spinner("Loading OncoScan AI model..."):

    try:
        model = load_oncoscan_model()

        st.success("Model loaded successfully!")

    except Exception as error:

        st.error(
            f"Failed to load model: {error}"
        )

        st.stop()


# ============================================================
# Display Classes
# ============================================================

st.write(
    "**Supported Classes:**",
    ", ".join(CLASS_NAMES)
)


# ============================================================
# Image Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input MRI Image",
        width=400
    )

    if st.button(
        "🔍 Analyze Image",
        type="primary"
    ):

        try:

            # ------------------------------------------------
            # Prepare image
            # ------------------------------------------------

            resized_image = image.resize(
                IMG_SIZE
            )

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # ------------------------------------------------
            # Generate Prediction + Grad-CAM
            # ------------------------------------------------

            with st.spinner(
                "Analyzing MRI image..."
            ):

                (
                    heatmap,
                    predicted_class,
                    probabilities
                ) = generate_gradcam_for_image(
                    model,
                    image_array
                )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            predicted_name = CLASS_NAMES[
                predicted_class
            ]

            confidence = (
                float(
                    probabilities[
                        predicted_class
                    ]
                ) * 100
            )

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            st.subheader(
                "Prediction Result"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Class",
                    predicted_name
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            # ------------------------------------------------
            # Probability Results
            # ------------------------------------------------

            st.subheader(
                "Class Probabilities"
            )

            class_probabilities = {}

            for index, class_name in enumerate(
                CLASS_NAMES
            ):

                probability = (
                    float(
                        probabilities[index]
                    ) * 100
                )

                class_probabilities[
                    class_name
                ] = probability

            for class_name, probability in (
                class_probabilities.items()
            ):

                st.write(
                    f"**{class_name}: "
                    f"{probability:.2f}%**"
                )

                st.progress(
                    min(
                        max(
                            probability / 100,
                            0.0
                        ),
                        1.0
                    )
                )

            # ------------------------------------------------
            # Grad-CAM
            # ------------------------------------------------

            st.subheader(
                "🔥 Grad-CAM Explanation"
            )

            import matplotlib.pyplot as plt

            original_array = np.array(
                resized_image
            )

            fig, ax = plt.subplots(
                figsize=(6, 6)
            )

            ax.imshow(
                original_array
            )

            ax.imshow(
                heatmap,
                cmap="jet",
                alpha=0.45
            )

            ax.set_title(
                f"Prediction: {predicted_name}\n"
                f"Confidence: {confidence:.2f}%"
            )

            ax.axis("off")

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.success(
                "Analysis completed successfully."
            )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )