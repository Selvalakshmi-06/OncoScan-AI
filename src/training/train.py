import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from src.preprocessing.data_loader import load_datasets, CLASS_NAMES

# ============================================================
# OncoScan AI - EfficientNetB0 Training
# ============================================================

IMG_SIZE = (224, 224)
NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = 15

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "oncoscan_efficientnetb0.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="data_augmentation")


# ============================================================
# Build model
# ============================================================

def build_model():

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained feature extractor initially
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)

    # EfficientNet includes its own input rescaling
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classification_output"
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Training
# ============================================================

def train():

    print("\nLoading datasets...")
    train_ds, val_ds, test_ds = load_datasets()

    print("\nBuilding EfficientNetB0 model...")
    model = build_model()

    model.summary()

    # Save the best validation model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Stop if validation performance stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate when validation loss stops improving
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print("\nStarting training...\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr
        ]
    )

    # Save training history
    with open(HISTORY_PATH, "w") as file:
        json.dump(history.history, file, indent=4)

    print("\nTraining completed!")
    print("Best model saved to:", MODEL_PATH)
    print("Training history saved to:", HISTORY_PATH)

    # Evaluate on the untouched test set
    print("\nEvaluating on the test set...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    print("\nFinal Test Accuracy:", test_accuracy)
    print("Final Test Loss:", test_loss)


if __name__ == "__main__":
    train()
