import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from src.preprocessing.data_loader import load_datasets, CLASS_NAMES

# ============================================================
# OncoScan AI - Model Evaluation
# ============================================================

MODEL_PATH = "models/oncoscan_efficientnetb0.keras"


def evaluate_model():

    print("\nLoading test dataset...")

    _, _, test_ds = load_datasets()

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nGenerating predictions...")

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(images, verbose=0)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)

        y_pred.extend(predicted_classes)
        y_true.extend(true_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ========================================================
    # Accuracy
    # ========================================================

    accuracy = accuracy_score(y_true, y_pred)

    print("\n========================================")
    print("MODEL EVALUATION RESULTS")
    print("========================================")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # ========================================================
    # Classification Report
    # ========================================================

    print("\nClassification Report:")
    print("----------------------------------------")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4
        )
    )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    print("Confusion Matrix:")
    print("----------------------------------------")

    cm = confusion_matrix(y_true, y_pred)

    print(cm)

    print("\nClass order:")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index}: {class_name}")


if __name__ == "__main__":
    evaluate_model()