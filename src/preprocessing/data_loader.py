import tensorflow as tf

# ============================================================
# OncoScan AI - Brain MRI Data Loader
# ============================================================

# Dataset paths
TRAIN_DIR = "data/raw/Training"
TEST_DIR = "data/raw/Testing"

# Image configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Class names
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


def load_datasets():
    """
    Load the training dataset and create a validation split.
    The testing dataset remains completely separate.
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.20,
        subset="training",
        seed=SEED
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.20,
        subset="validation",
        seed=SEED
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Improve input pipeline performance
    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
    test_dataset = test_dataset.prefetch(buffer_size=autotune)

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_datasets()

    print("\nDataset loading successful!")
    print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Testing batches:", tf.data.experimental.cardinality(test_ds).numpy())
    print("Classes:", CLASS_NAMES)
