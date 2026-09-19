# models/01_baseline_cnn.py

import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(num_classes=10):
    """
    Builds a small, deliberately simple CNN for Fashion-MNIST.

    Unlike the underwater project (where architecture choices were driven
    by "what gets the best accuracy"), every choice here is driven by
    "what keeps this network easy to visualize and understand" -- small
    filter counts, a shallow depth, and an architecture that ends in
    GlobalAveragePooling2D -> Dense, which is the same design used in the
    original Class Activation Mapping paper (the basis for Grad-CAM, which
    we'll implement in Step 8).
    """

    model = models.Sequential(name="Interpretable_Fashion_CNN")

    # Explicit Input layer: tells Keras the shape of one single image
    # (28 height, 28 width, 1 channel -- grayscale). This must match
    # exactly what load_and_preprocess_data() produces.
    model.add(layers.Input(shape=(28, 28, 1)))

    # --- Conv block 1 ---
    # Conv2D with only 8 filters (small on purpose -- easy to visualize
    # all 8 filters at once later). 3x3 is the standard small kernel size,
    # padding='same' keeps the output the same width/height as the input
    # (28x28 stays 28x28 here), which keeps spatial positions easy to
    # reason about between layers.
    model.add(layers.Conv2D(8, (3, 3), padding="same", activation="relu"))

    # BatchNormalization rescales this layer's outputs to a stable range,
    # which helps training converge faster and more reliably. We're
    # keeping this even though our dataset is large (60,000 images),
    # since it helps regardless of dataset size and doesn't complicate
    # interpretation the way Dropout's randomness would.
    model.add(layers.BatchNormalization())

    # MaxPooling2D with a 2x2 window halves height and width:
    # 28x28 -> 14x14. The 8 channels/filters stay unchanged -- pooling
    # only shrinks spatial size, never depth.
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Conv block 2 ---
    # Filters increase 8 -> 16 as is conventional (deeper layers usually
    # need more filters to represent more complex combinations of the
    # simpler patterns detected earlier), while still staying small
    # enough to visualize comfortably.
    model.add(layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    # 14x14 -> 7x7
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Conv block 3 ---
    # No pooling after this block -- we stop shrinking here so the final
    # feature maps (7x7) are still large enough to produce a meaningful
    # Grad-CAM heatmap later. Shrinking further (e.g. to 3x3 or 1x1)
    # would make that heatmap too coarse to be useful.
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())

    # GlobalAveragePooling2D collapses each of the 32 feature maps
    # (each 7x7) down to a single average value, producing a flat
    # vector of just 32 numbers. This is deliberately used instead of
    # Flatten() here -- not because Flatten would be unsafe on this
    # dataset size (it wouldn't be), but because GlobalAveragePooling2D
    # -> Dense is the exact architecture pattern that makes Grad-CAM
    # work cleanly in Step 8.
    model.add(layers.GlobalAveragePooling2D())

    # Final output layer: one output per class (10 for Fashion-MNIST),
    # softmax activation converts raw scores into probabilities that
    # sum to 1 across all 10 classes.
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Note: we deliberately do NOT call model.compile() here.
    # Compilation (choosing optimizer, loss function, metrics) is a
    # training-time decision, not an architecture decision -- keeping
    # them separate means this function's only job is "define the
    # shape of the network," which keeps it reusable and easy to test
    # on its own.
    return model


if __name__ == "__main__":
    # Quick standalone check: build the model and print its summary,
    # so we can visually confirm the layer shapes and parameter counts
    # before writing any training code that depends on this function.
    model = build_baseline_cnn()
    model.summary()