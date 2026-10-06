import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.preprocessing.data_loader import CLASS_NAMES

# ============================================================
# OncoScan AI - Grad-CAM Explainability
# ============================================================

MODEL_PATH = "models/oncoscan_efficientnetb0.keras"
TEST_DIR = "data/raw/Testing"

IMG_SIZE = (224, 224)

OUTPUT_DIR = "outputs/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    print("\nLoading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    return model


def find_test_images():
    """Find one test image from each class."""

    test_images = []

    for class_name in CLASS_NAMES:

        class_dir = os.path.join(
            TEST_DIR,
            class_name
        )

        if not os.path.exists(class_dir):
            print(f"Warning: folder not found: {class_dir}")
            continue

        for filename in sorted(os.listdir(class_dir)):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                image_path = os.path.join(
                    class_dir,
                    filename
                )

                test_images.append(
                    (class_name, image_path)
                )

                break

    if not test_images:
        raise FileNotFoundError(
            "No test images found."
        )

    return test_images


def prepare_image(image_path):
    """Load and prepare image for the model."""

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def make_gradcam_heatmap(model, image_array):
    """Generate Grad-CAM heatmap."""

    base_model = model.get_layer(
        "efficientnetb0"
    )

    last_conv_layer = base_model.get_layer(
        "top_conv"
    )

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    augmentation_layer = model.get_layer(
        "data_augmentation"
    )

    augmented_image = augmentation_layer(
        image_array,
        training=False
    )

    with tf.GradientTape() as tape:

        conv_outputs, base_output = grad_model(
            augmented_image,
            training=False
        )

        x = model.get_layer(
            "global_average_pooling2d"
        )(base_output)

        x = model.get_layer(
            "dropout"
        )(x)

        predictions = model.get_layer(
            "classification_output"
        )(x)

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_output = predictions[
            :,
            predicted_class
        ]

    gradients = tape.gradient(
        class_output,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if max_value > 0:
        heatmap /= max_value

    return (
        heatmap.numpy(),
        int(predicted_class.numpy()),
        predictions.numpy()[0]
    )


def save_gradcam_result(
    image_path,
    heatmap,
    predicted_class,
    probabilities,
    true_class
):
    """Save original image and Grad-CAM result."""

    original_image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    original_array = np.array(
        original_image
    )

    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    confidence = (
        probabilities[predicted_class]
        * 100
    )

    plt.figure(figsize=(10, 4))

    # Original image
    plt.subplot(1, 2, 1)

    plt.imshow(
        original_array
    )

    plt.title(
        f"Original MRI\n"
        f"True class: {true_class}"
    )

    plt.axis("off")

    # Grad-CAM
    plt.subplot(1, 2, 2)

    plt.imshow(
        original_array
    )

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"Grad-CAM\n"
        f"Prediction: {predicted_name}\n"
        f"Confidence: {confidence:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    output_filename = (
        f"{true_class}_gradcam.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def main():

    print("\n========================================")
    print("       ONCOSCAN AI - GRAD-CAM")
    print("========================================")

    # Load trained model
    model = load_model()

    # Find one image from every class
    test_images = find_test_images()

    print(
        f"\nFound {len(test_images)} test images."
    )

    # Process each class
    for true_class, image_path in test_images:

        print("\n========================================")
        print(
            f"Testing class: {true_class}"
        )
        print("========================================")

        print("\nImage:")
        print(image_path)

        # Prepare image
        image_array = prepare_image(
            image_path
        )

        # Generate Grad-CAM
        print(
            "\nGenerating Grad-CAM heatmap..."
        )

        (
            heatmap,
            predicted_class,
            probabilities
        ) = make_gradcam_heatmap(
            model,
            image_array
        )

        predicted_name = CLASS_NAMES[
            predicted_class
        ]

        confidence = (
            probabilities[predicted_class]
            * 100
        )

        print("\nTrue class:", true_class)

        print(
            "Predicted class:",
            predicted_name
        )

        print(
            f"Prediction confidence: "
            f"{confidence:.2f}%"
        )

        print(
            "\nClass probabilities:"
        )

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):
            print(
                f"{class_name}: "
                f"{probability * 100:.2f}%"
            )

        # Save result
        output_path = save_gradcam_result(
            image_path,
            heatmap,
            predicted_class,
            probabilities,
            true_class
        )

        print(
            "\nGrad-CAM image saved to:"
        )

        print(output_path)

    print("\n========================================")
    print(
        "All Grad-CAM tests completed!"
    )
    print("========================================")
def generate_gradcam_for_image(model, image_array):
    """
    Generate Grad-CAM heatmap for an uploaded image.

    Returns:
        heatmap
        predicted_class
        probabilities
    """

    heatmap, predicted_class, probabilities = make_gradcam_heatmap(
        model,
        image_array
    )

    return (
        heatmap,
        predicted_class,
        probabilities
    )

if __name__ == "__main__":
    main()