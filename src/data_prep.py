import tensorflow as tf
import numpy as np

# Class names aren't included in the dataset itself -- Fashion-MNIST only
# gives you integer labels 0-9, so we define what each number means here.
# We'll reuse this constant everywhere we need to display a readable label.
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_and_preprocess_data():
    """
    Loads Fashion-MNIST and returns fully preprocessed train/test data,
    ready to feed directly into a Conv2D-based model.
    """

    # Step 1: Load the raw data.
    # Keras bundles this dataset directly, so no manual download or
    # train/test splitting is needed -- it comes pre-split.
    # Raw shape at this point: X_train is (60000, 28, 28), values 0-255 (uint8)
    (X_train,y_train), (X_test , y_test)=tf.keras.datasets.fashion_mnist.load_data()
    
    print(f"Raw shapes -- X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Raw dtype: {X_train.dtype}, raw value range: {X_train.min()}-{X_train.max()}")
    
    # Step 2: Normalize pixel values.
    # Convert to float32 FIRST, then divide -- dividing uint8 integers by 255
    # would silently produce 0s for every value except 255 itself, since
    # integer division truncates decimals. Casting to float32 avoids this trap.
    
    X_train=X_train.astype("float32")/ 255.0
    X_test=X_test.astype("float32")/255.0
    print(f"After normalization -- dtype: {X_train.dtype}, value range: {X_train.min()}-{X_train.max()}")
    
    # Step 3: Add the channel dimension.
    # Conv2D layers expect shape (height, width, channels), even for
    # grayscale images where channels=1. Right now our images are just
    # (28, 28) with no channel dimension at all, so we add one explicitly.
    # np.newaxis inserts a new axis of size 1 at the end.
    X_train=X_train[..., np.newaxis]
    X_test=X_test[... , np.newaxis]
    print(f"After reshaping -- X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Step 4: Return everything the rest of the project will need.
    # Labels are left as plain integers (0-9) rather than one-hot encoded,
    # since we'll use sparse_categorical_crossentropy as our loss function,
    # which expects integer labels directly.
    return X_train , y_train ,X_test , y_test

if __name__ == "__main__":
    # Quick standalone check: running this file directly lets you verify
    # the function works correctly before any other script depends on it.
    X_train , y_train ,X_test , y_test=load_and_preprocess_data()
    
    print("\nFinal check:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"Example label: {y_train[0]} -> {CLASS_NAMES[y_train[0]]}")
    

