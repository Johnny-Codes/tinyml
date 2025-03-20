import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the model
model_path = (
    "../models/casting/mobilenetv3_small/full/mobilenetv3_small-1-t32-v16-0.9986.pth"
)
model = models.mobilenet_v3_small()
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)  # Adjust the classifier layer

# Load the model state dict with map_location to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# Preprocess the input image
def preprocess_image(image_path):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    return image


# Hook for Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0] if grad_out else None

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, target_class]
        target.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
        cam = np.uint8(cam * 255)
        cam = np.uint8(Image.fromarray(cam).resize((224, 224), Image.LANCZOS)) / 255

        return cam


# Plot and save saliency maps in a grid
def plot_saliency_maps(model, image_paths, output_path):
    # Collect layers, excluding specific layers
    layers = [
        module for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    ]

    # Exclude specific layers by their indices or other criteria
    layers_to_exclude = [
        2,
        3,
        13,
        14,
        18,
        19,
        23,
        24,
        28,
        29,
        33,
        34,
        38,
        39,
        43,
        44,
        48,
        49,
    ]  # Example: Exclude the first two layers
    layers = [layer for i, layer in enumerate(layers) if i not in layers_to_exclude]

    # Use only the last layer
    last_layer = layers[-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()  # Flatten the 2x2 array of axes for easy indexing

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        image = image.resize(
            (224, 224)
        )  # Resize the original image to match the CAM size
        image_tensor = preprocess_image(image_path).to(device)

        # Plot original image
        axes[i].imshow(image)
        axes[i].axis("off")
        image_class = image_path.split("/")[-2]
        image_name = ""
        if image_class == "ok_front":
            image_name = "Non-defective"
        else:
            image_name = "Defective"
        axes[i].set_title(
            f"{image_name}", y=1.0, fontsize=16
        )  # Class label, title above image

        # Generate CAM for the last layer
        grad_cam = GradCAM(model, last_layer)
        cam = grad_cam.generate_cam(
            image_tensor, target_class=0
        )  # Assuming class 0 for visualization

        # Reverse the color map
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # heatmap = cv2.bitwise_not(heatmap)  # Reverse the color map
        heatmap = np.float32(heatmap) / 255
        cam_image = heatmap + np.float32(image) / 255
        cam_image = cam_image / np.max(cam_image)

        # Plot the CAM image
        axes[i + 2].imshow(cam_image)
        axes[i + 2].axis("off")
        # axes[i + 2].set_title("Grad-CAM", y=0.0)  # Title above image

    # Add a color bar as a legend
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="jet_r"),
        ax=axes,
        orientation="horizontal",
        fraction=0.05,  # Adjust fraction to control the size of the color bar
        pad=0.05,  # Adjust pad to control the spacing between the color bar and the plot
    )
    cbar.ax.tick_params(labelsize=14, colors="black")
    cbar.ax.set_position(
        [0.333333334, 0.24825, 0.333333334, 0.25]  # [left, bottom, width, height]
    )

    cbar.set_label(
        "Grad-CAM Intensity",
        fontsize=18,
        color="black",
    )

    plt.tight_layout()

    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    image_paths = [
        "../casting_data/casting_data/g_val/def_front/cast_def_0_350.jpeg",
        "../casting_data/casting_data/g_val/ok_front/cast_ok_0_307.jpeg",
    ]  # Replace with your image paths
    output_path = "updated_saliency_maps_grid_output_2x2.png"
    # Replace with your desired output path
    plot_saliency_maps(model, image_paths, output_path)
