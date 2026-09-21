# src/visualize.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_conv_layer_names(model):
    """
    Returns the names of every Conv2D layer in the model, in order.
    We need this because we'll want to visualize filters from EACH
    conv layer (layer 1, 2, and 3), and layer names are how we tell
    Keras which layer's weights we want to inspect.
    """
    conv_layer_names = []
    for layer in model.layers:
        # __class__.__name__ gives us the layer type as a string,
        # e.g. "Conv2D", "BatchNormalization", "Dense" -- this lets
        # us filter for only the conv layers without hardcoding names.
        if layer.__class__.__name__ == "Conv2D":
            conv_layer_names.append(layer.name)
    return conv_layer_names


def visualize_filters(model, layer_name, save_path=None):
    """
    Plots every filter from one named Conv2D layer as small grayscale
    images.

    For the FIRST conv layer (1 input channel), each filter is a single
    3x3 image -- straightforward to interpret directly (edges, gradients).

    For DEEPER conv layers (multiple input channels), each "filter" is
    actually a stack of one 3x3 grid PER input channel. We plot this as
    a grid: one row per output filter, one column per input channel --
    this visually demonstrates that a deeper filter combines information
    across every channel the previous layer produced, not just one.
    """
    layer = model.get_layer(layer_name)

    # get_weights() returns a list: [kernel_weights, bias_weights].
    # We only want the kernel (the actual filter grids), not the bias.
    # Shape is (kernel_height, kernel_width, input_channels, output_filters).
    filters, biases = layer.get_weights()

    kernel_h, kernel_w, input_channels, output_filters = filters.shape
    print(f"Layer '{layer_name}': {output_filters} filters, each {kernel_h}x{kernel_w}, "
          f"operating over {input_channels} input channel(s)")

    # Normalize all filter values to a 0-1 range for consistent, readable
    # display. Raw weights can be small positive/negative floats that
    # would otherwise render as a nearly uniform gray image -- min-max
    # scaling stretches them to use the full black-to-white range.
    f_min, f_max = filters.min(), filters.max()
    filters_normalized = (filters - f_min) / (f_max - f_min + 1e-8)

    # Build a grid: rows = output filters, columns = input channels.
    # For layer 1, this is simply (8 rows, 1 column) -- one image per filter.
    # For layer 2, this is (16 rows, 8 columns) -- showing exactly how
    # each filter is built from a slice per input channel.
    fig, axes = plt.subplots(
        output_filters, input_channels,
        figsize=(max(input_channels * 1.2, 3), output_filters * 1.2)
    )

    # When there's only 1 input channel, matplotlib gives us a 1D array
    # of axes instead of 2D -- reshape so our indexing logic below works
    # the same way regardless of input_channels.
    axes = np.array(axes).reshape(output_filters, input_channels)

    for out_idx in range(output_filters):
        for in_idx in range(input_channels):
            ax = axes[out_idx, in_idx]
            # Slice out this specific (input_channel, output_filter) grid.
            single_filter_slice = filters_normalized[:, :, in_idx, out_idx]
            ax.imshow(single_filter_slice, cmap="gray")
            ax.axis("off")
            if out_idx == 0:
                ax.set_title(f"in ch {in_idx}", fontsize=8)
            if in_idx == 0:
                ax.set_ylabel(f"filter {out_idx}", fontsize=8, rotation=0, ha="right", va="center")

    plt.suptitle(f"Filters -- {layer_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved filter visualization to {save_path}")
    plt.show()
    

# Add this to src/visualize.py (alongside get_conv_layer_names and visualize_filters)

def visualize_feature_maps(model, image, conv_layer_names, save_path=None):
    """
    Feeds a single image through the model and displays the feature maps
    (intermediate activations) produced at each named Conv2D layer, along
    with a printed explanation of what that layer's position in the
    network implies about what it's likely detecting.
    """
    layer_outputs = [model.get_layer(name).output for name in conv_layer_names]
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)

    image_batch = np.expand_dims(image, axis=0)
    activations = activation_model.predict(image_batch, verbose=0)

    plt.figure(figsize=(3, 3))
    plt.imshow(image[:, :, 0], cmap="gray")
    plt.title("Original input image")
    plt.axis("off")
    plt.show()

    # Short, position-based explanations -- first layer sees the raw
    # image directly, the last layer is the most removed from it, and
    # anything in between is transitioning from concrete to abstract.
    num_layers = len(conv_layer_names)

    for idx, (layer_name, activation) in enumerate(zip(conv_layer_names, activations)):
        num_filters = activation.shape[-1]

        # Compute how many of this layer's feature maps are nearly
        # inactive for THIS specific image -- a genuinely useful,
        # image-specific observation rather than generic boilerplate.
        # We call a feature map "inactive" if its highest activation
        # value is very close to zero (ReLU outputs are always >= 0,
        # so near-zero means the filter barely fired at all).
        max_per_filter = activation[0].max(axis=(0, 1))
        inactive_count = int(np.sum(max_per_filter < 0.05))

        # Print a plain-language explanation based on this layer's
        # depth in the network.
        print(f"\n--- {layer_name} ({num_filters} filters) ---")
        if idx == 0:
            print("This is the FIRST conv layer -- it sees the raw image directly.")
            print("Expect these feature maps to still resemble the original shape,")
            print("since each filter is just responding to simple patterns like edges.")
        elif idx == num_layers - 1:
            print("This is the DEEPEST conv layer -- it never sees the raw image,")
            print("only combinations of the previous layer's feature maps.")
            print("Expect these to look abstract and hard to describe visually.")
        else:
            print("This is a MIDDLE conv layer -- it combines the previous layer's")
            print("simple patterns into more complex, but still developing, shapes.")

        print(f"{inactive_count}/{num_filters} filters barely activated for this image "
              f"(may be specialized for patterns this image doesn't contain).")

        num_cols = min(8, num_filters)
        num_rows = int(np.ceil(num_filters / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))
        axes = np.array(axes).reshape(num_rows, num_cols)

        for i in range(num_rows * num_cols):
            ax = axes[i // num_cols, i % num_cols]
            if i < num_filters:
                ax.imshow(activation[0, :, :, i], cmap="viridis")
                ax.set_title(f"filter {i}", fontsize=7)
            ax.axis("off")

        plt.suptitle(f"Feature maps -- {layer_name} ({num_filters} filters)")
        plt.tight_layout()

        if save_path:
            fig.savefig(f"{save_path}_{layer_name}.png")
            print(f"Saved feature maps to {save_path}_{layer_name}.png")
        plt.show()