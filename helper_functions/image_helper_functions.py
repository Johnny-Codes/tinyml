import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from torchvision import transforms
from sklearn.metrics import confusion_matrix


def save_plot(data, title, ylabel, xlabel, filename):
    """
    create a save a plot based on the provided data

    Args:
        data (iterable): Data points to be plotted.
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
        xlabel (str): Label for the x-axis.
        filename (str): The name of the file where the plot will be saved.

    The plot is generated using Seaborn's 'whitegrid' style and saved to the specified filename
    """
    print(f"Saving graph {title} to {filename}")
    plt.figure()
    sns.set_theme(style="whitegrid")
    plt.plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename, dpi=600)
    plt.close()


def save_sample_images(inputs, labels, preds, class_names, filename):
    """
    Save a grid of sample images with predicted and true class labels.

    Args:
        inputs (torch.Tensor): Batch of input images.
        labels (torch.Tensor): True labels of the input images.
        preds (torch.Tensor): Predicted labels of the input images.
        class_names (list): List of class names corresponding to label indices.
        filename (str): The filename where the image grid will be saved.
    """
    print(f"Saving {filename}")
    inputs = inputs.cpu()
    labels = labels.cpu()
    preds = preds.cpu()
    num_images = min(len(inputs), 16)  # Limit to 16 images
    fig = plt.figure(figsize=(15, 15))
    for i in range(num_images):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        img = transforms.ToPILImage()(inputs[i])
        ax.imshow(img)

        # Check if prediction matches the actual label
        if preds[i] == labels[i]:
            color = "green"
        else:
            color = "red"

        # Create a rectangle patch with the background color
        rect = patches.Rectangle(
            (0, 0), 224, 224, linewidth=0, edgecolor="none", facecolor=color, alpha=0.3
        )
        ax.add_patch(rect)

        # Set the title with white bold font
        ax.set_title(
            f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}",
            color="white",
            fontweight="bold",
        )

    plt.savefig(filename, dpi=600)
    plt.close()


def visualize_feature_maps(model, layer_name, input_image, device):
    """
    Visualize and save feature maps from a specific layer of a model given an input image.

    Args:
        model (torch.nn.Module): The neural network model containing the layer.
        layer_name (str): The name of the layer from which to extract the feature maps.
        input_image (torch.Tensor): The input image for which the feature maps will be computed.
        device (torch.device): The device on which the model and input image are located.

    This function registers a forward hook on the specified layer to capture the output feature maps.
    It runs the model in evaluation mode, processes the input image through the model, and extracts
    the feature maps. The feature maps are then visualized and saved as a PNG file named
    `feature_maps_{layer_name}.png`.
    """
    print(f"Visualizing feature maps for layer: {layer_name}")

    def hook_fn(module, input, output):
        feature_maps.append(output)

    feature_maps = []
    layer = dict([*model.named_modules()])[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        input_image = input_image.unsqueeze(0).to(device)
        model(input_image)

    hook.remove()

    feature_maps = feature_maps[0].cpu().numpy()
    num_feature_maps = feature_maps.shape[1]

    fig, axes = plt.subplots(1, num_feature_maps, figsize=(30, 30))
    for i in range(num_feature_maps):
        axes[i].imshow(feature_maps[0, i], cmap="viridis")
        axes[i].axis("off")
    plt.savefig(f"feature_maps_{layer_name}.png", dpi=600)
    plt.close()


def visualize_feature_maps_per_layer(model, input_image, device):
    model.eval()  # Set the model to evaluation mode
    input_image = input_image.unsqueeze(0).to(
        device
    )  # Add batch dimension and move to device

    # Create directory to save images
    os.makedirs("img_ftr_layers", exist_ok=True)

    # Hook to extract the feature maps
    def hook_fn(module, input, output, layer_name):
        feature_maps = output.detach().cpu()
        num_feature_maps = feature_maps.shape[1]
        grid_size = int(np.ceil(np.sqrt(num_feature_maps)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i in range(num_feature_maps):
            ax = axes[i]
            ax.imshow(
                feature_maps[0, i], cmap="gray", alpha=0.5
            )  # Set alpha for transparency
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_facecolor("none")  # Set face color to none for transparency

        # Hide any unused subplots
        for i in range(num_feature_maps, len(axes)):
            axes[i].axis("off")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(
            f"img_ftr_layers/feature_maps_{layer_name}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,  # Save the figure with a transparent background
        )
        plt.close(fig)

    # Register hooks for all Conv2d layers
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):  # Only visualize Conv2d layers
            hooks.append(
                layer.register_forward_hook(
                    lambda module, input, output, name=name: hook_fn(
                        module, input, output, name
                    )
                )
            )

    # Forward pass through the model
    with torch.no_grad():
        model(input_image)

    # Remove all hooks
    for hook in hooks:
        hook.remove()


def visualize_feature_maps_black_bg(model, input_image, layer_name):
    def hook_fn(module, input, output, name):
        feature_maps = output.detach().cpu().numpy()
        num_feature_maps = feature_maps.shape[1]
        grid_size = int(np.ceil(np.sqrt(num_feature_maps)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
        axes = axes.flatten()
        for i in range(num_feature_maps):
            ax = axes[i]
            ax.imshow(feature_maps[0, i], cmap="gray")
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_facecolor("black")

        # Hide any unused subplots
        for i in range(num_feature_maps, len(axes)):
            axes[i].axis("off")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(
            f"img_ftr_layers/feature_maps_black_bg_{layer_name}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

    # Register hooks for the specified layer
    hooks = []
    for name, layer in model.named_modules():
        if name == layer_name:
            hooks.append(
                layer.register_forward_hook(
                    lambda module, input, output, name=name: hook_fn(
                        module, input, output, name
                    )
                )
            )

    # Forward pass through the model
    with torch.no_grad():
        model(input_image)

    # Remove all hooks
    for hook in hooks:
        hook.remove()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    """
    Plot and save a confusion matrix.

    Args:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.
        class_names (list): List of class names corresponding to label indices.
        filename (str): The filename where the confusion matrix will be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(filename, dpi=600)
    plt.close()
