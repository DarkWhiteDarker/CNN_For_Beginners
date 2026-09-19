# src/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, class_names, save_path=None):
    """
    Evaluates a trained model on the test set: prints loss/accuracy,
    a full classification report, and plots a confusion matrix.
    This is the same shared evaluation function pattern used in the
    underwater project -- reused here so both projects report results
    consistently.
    """
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    # predict() returns probabilities for all 10 classes per image;
    # argmax picks the class with the highest probability as the
    # model's actual predicted label.
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Test Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()

    return test_acc


def plot_training_curves(history, title_suffix="", save_path=None):
    """
    Plots training vs validation accuracy and loss over epochs,
    so we can visually check for overfitting (a growing gap between
    the two lines) the same way we did for the underwater project.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="o")
    plt.title(f"Accuracy {title_suffix}")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="o")
    plt.title(f"Loss {title_suffix}")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved training curves to {save_path}")
    plt.show()