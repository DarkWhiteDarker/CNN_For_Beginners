# models/03_class_weighted_cnn.py

import sys, os
sys.path.append(os.path.abspath("."))

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from src.data_prep import load_and_preprocess_data, CLASS_NAMES
from src.evaluate import evaluate_model, plot_training_curves


def build_baseline_cnn(num_classes=10):
    """Identical to the Step 4/5 baseline architecture -- copied here so
    this file can run standalone without fighting Python's import rules
    on a filename starting with a digit."""
    model = models.Sequential(name="Interpretable_Fashion_CNN")
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    model = build_baseline_cnn()
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = {i: 1.0 for i in range(10)}
    class_weights[6] = 3.0
    print(f"Using class weights: {class_weights}")

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        batch_size=32,
        epochs=15,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1,
    )

    plot_training_curves(history, "(Class-Weighted CNN)", save_path="reports/figures/class_weighted_curves.png")
    evaluate_model(model, X_test, y_test, CLASS_NAMES, save_path="reports/figures/class_weighted_confusion.png")

    model.save("models/class_weighted_cnn.keras")
    print("Saved model to models/class_weighted_cnn.keras")


if __name__ == "__main__":
    main()