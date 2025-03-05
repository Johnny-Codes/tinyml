"""
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_combined_weight_distribution(model, model_name):
    conv_weights = []
    fc_weights = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.data.cpu().numpy().flatten()
            if "conv" in name:
                conv_weights.extend(weights)
            elif "fc" in name or "classifier" in name:
                fc_weights.extend(weights)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].hist(conv_weights, bins=50, alpha=0.75, color="blue")
    axes[0].set_title("Convolutional Layers")
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    axes[1].hist(fc_weights, bins=50, alpha=0.75, color="green")
    axes[1].set_title("Fully Connected Layers")
    axes[1].set_xlabel("Weight")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_box_plots(model, model_name):
    layer_weights = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.data.cpu().numpy().flatten()
            layer_weights.append(weights)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.boxplot(layer_weights, vert=False, patch_artist=True)
    ax.set_yticklabels(
        [name for name, _ in model.named_parameters() if _.requires_grad]
    )
    ax.set_title("Weight Distributions by Layer")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    plt.show()


def load_model(model_path, model_architecture, device):
    model = model_architecture
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model architecture
    from torchvision.models import mobilenet_v2

    model_architecture = mobilenet_v2(weights=None)

    # Load the model
    model_path = "./models/casting/mobilenetv2/full/mobilenetv2-0-t32-v16-0.9972.pth"
    model_name = "mobilenetv2"
    model = load_model(model_path, model_architecture, device)

    # Plot combined weight distribution
    plot_combined_weight_distribution(model, model_name)

    # Plot box plots of weight distributions
    plot_box_plots(model, model_name)
"""

"""import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_key_layer_weight_distribution(model, model_name):
    key_layers = [
        "features.0",
        "features.1",
        "features.18",
        "classifier.0",
        "classifier.1",
    ]
    layer_weights = {layer: [] for layer in key_layers}

    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.data.cpu().numpy().flatten()
            for layer in key_layers:
                if layer in name:
                    layer_weights[layer].extend(weights)

    fig, axes = plt.subplots(1, len(key_layers), figsize=(20, 6))

    for idx, (layer, weights) in enumerate(layer_weights.items()):
        axes[idx].hist(weights, bins=50, alpha=0.75, color="blue")
        axes[idx].set_title(f"{layer}")
        axes[idx].set_xlabel("Weight")
        axes[idx].set_ylabel("Frequency")
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()


def plot_box_plots(model, model_name):
    key_layers = [
        "features.0",
        "features.1",
        "features.18",
        "classifier.0",
        "classifier.1",
    ]
    layer_weights = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.data.cpu().numpy().flatten()
            for layer in key_layers:
                if layer in name:
                    layer_weights.append(weights)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.boxplot(layer_weights, vert=False, patch_artist=True)
    ax.set_yticklabels(key_layers)
    ax.set_title("Weight Distributions by Key Layers")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    plt.show()


def load_model(model_path, model_architecture, device):
    model = model_architecture
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model architecture
    from torchvision.models import mobilenet_v2

    model_architecture = mobilenet_v2(weights=None)

    # Load the model
    model_path = "./models/casting/mobilenetv2/full/mobilenetv2-0-t32-v16-0.9972.pth"
    model_name = "mobilenetv2"
    model = load_model(model_path, model_architecture, device)

    # Plot weight distribution for key layers
    plot_key_layer_weight_distribution(model, model_name)

    # Plot box plots of weight distributions for key layers
    plot_box_plots(model, model_name)
"""
