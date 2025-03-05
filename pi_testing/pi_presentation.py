import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import tkinter

# Set the quantization engine to qnnpack
# torch.backends.quantized.engine = "qnnpack"

# Define the image transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Load the dataset
data_dir = "../casting_data/casting_data/g_test"
# data_dir = "../g_test_casting"
dataset = ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model Path (You can change this)
MODEL_PATH = (
    "../models/casting/mobilenetv3_small/full/mobilenetv3_small-1-t32-v16-0.9986.pth"
)


def get_model_architecture(model_name="mobilenet_v3_small"):
    if model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


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

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0] if len(grad_output) > 0 else None

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


def display_image(image, cam=None, prediction=None, true_label=None):
    """Displays the image, Grad-CAM, and classification result in fullscreen."""
    plt.close("all")  # Close all existing figures

    # Get screen resolution
    try:
        root = tkinter.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()  # Close the Tkinter window
    except tkinter.TclError:
        print("Unable to determine screen resolution. Using default size.")
        screen_width = 1920  # Default width
        screen_height = 1080  # Default height

    # Calculate figure size based on screen resolution
    fig_width = screen_width / 100  # Adjust scaling factor as needed
    fig_height = screen_height / 100  # Adjust scaling factor as needed

    fig, axs = plt.subplots(
        1, 2, figsize=(fig_width, fig_height)
    )  # Create a figure and a set of subplots

    # Display the original image
    axs[0].imshow(image)
    axs[0].axis("off")
    axs[0].set_title("Original Image", size=36)

    # Add the true label text
    true_label_text = "Non-Defective" if true_label == 1 else "Defective"
    axs[0].text(
        0.5,
        -0.1,
        f"True: {true_label_text}",
        size=36,
        ha="center",
        transform=axs[0].transAxes,
    )

    if cam is not None:
        # Define the custom colormap
        colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Red, Yellow, Green
        cmap_name = "ryg"
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

        # Overlay the CAM on the original image
        heatmap = custom_cmap(cam)
        # heatmap = np.float32(heatmap)
        heatmap = np.float32(heatmap[:, :, :3])  # Sliced to remove the alpha channel

        # Set transparency (alpha) of the heatmap
        alpha = 0.6  # Adjust as needed

        # Overlay the CAM on the original image with transparency
        cam_image = alpha * heatmap + (1 - alpha) * image
        # cam_image = alpha * heatmap + (1 - alpha) * image / 255
        # cam_image = heatmap +  image / 255

        cam_image = cam_image / np.max(cam_image)

        # Display the CAM image
        im = axs[1].imshow(cam_image, cmap=custom_cmap)  # Specify the colormap here
        axs[1].axis("off")
        axs[1].set_title("Grad-CAM", size=36)

        # Add the predicted label text
        predicted_label_text = "Non-Defective" if prediction == 1 else "Defective"
        axs[1].text(
            0.5,
            -0.1,
            f"Predicted: {predicted_label_text}",
            size=36,
            ha="center",
            transform=axs[1].transAxes,
        )

        # Add colorbar
        norm = Normalize(vmin=0, vmax=1)
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=axs[1], shrink=0.8
        )
        cbar.ax.set_ylabel("Activation", rotation=-90, va="bottom", size=36)

    else:
        axs[1].axis("off")

    # Add a green or red box based on the prediction
    if prediction is not None:
        color = "green" if prediction == 1 else "red"  # Assuming 0 is non-defective
        for spine in axs[0].spines.values():
            spine.set_color(color)
            spine.set_linewidth(4)

    # Attempt to toggle fullscreen
    try:
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()
    except AttributeError:
        print("Fullscreen not supported in this environment.")

    plt.pause(10)  # Pause briefly to allow the plot to update
    plt.show(block=False)  # Show the plot without blocking execution


# Load the model
model = get_model_architecture()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Get the last convolutional layer
layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
last_layer = layers[-1]

grad_cam = GradCAM(model, last_layer)

for images, labels in dataloader:
    # Convert image tensor to PIL Image for display
    img = images.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    predicted_label = preds.cpu().numpy()[0]

    # Generate Grad-CAM
    cam = grad_cam.generate_cam(images, target_class=0)

    # Display the image, Grad-CAM, and classification result
    display_image(
        img, cam=cam, prediction=predicted_label, true_label=labels.cpu().numpy()[0]
    )

plt.close("all")
