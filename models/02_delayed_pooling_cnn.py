# models/02_delayed_pooling_cnn.py

import sys, os
sys.path.append(os.path.abspath("."))

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from src.data_prep import load_and_preprocess_data, CLASS_NAMES
from src.evaluate import evaluate_model, plot_training_curves


def build_delayed_pooling_cnn(num_classes=10):
    """
    Same overall design philosophy as the baseline, but with ONE change:
    an extra Conv2D(8) layer added BEFORE the first pooling operation.

    Hypothesis being tested: the baseline's Shirt misclassifications may
    be caused by fine details (like buttons) being lost too early, since
    the original baseline pools immediately after the very first conv
    layer. Keeping the image at full 28x28 resolution through TWO conv
    layers, instead of one, might let the network build slightly richer
    early representations before that detail is discarded.
    """
    model = models.Sequential(name="Delayed_Pooling_CNN")
    model.add(layers.Input(shape=(28, 28, 1)))

    # --- Conv block 1a (unchanged from baseline) ---
    model.add(layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())

    # --- Conv block 1b (NEW -- the actual change being tested) ---
    # Still 28x28 resolution here, still 8 filters -- this layer gets a
    # "second look" at the full-resolution image before anything shrinks.
    model.add(layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())

    # NOW we pool, one layer later than the baseline did.
    model.add(layers.MaxPooling2D((2, 2)))  # 28x28 -> 14x14

    # --- Conv block 2 (same as baseline from here on) ---
    model.add(layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))  # 14x14 -> 7x7

    # --- Conv block 3 (same as baseline) ---
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    model = build_delayed_pooling_cnn()
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        batch_size=32,
        epochs=15,
        callbacks=[early_stop],
        verbose=1,
    )

    plot_training_curves(history, "(Delayed Pooling CNN)", save_path="reports/figures/delayed_pooling_curves.png")
    evaluate_model(model, X_test, y_test, CLASS_NAMES, save_path="reports/figures/delayed_pooling_confusion.png")

    model.save("models/delayed_pooling_cnn.keras")
    print("Saved model to models/delayed_pooling_cnn.keras")


if __name__ == "__main__":
    main()