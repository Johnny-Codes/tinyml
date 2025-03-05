import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the model
model_path = (
    "./models/casting/mobilenetv3_small/full/mobilenetv3_small-1-t32-v16-0.9986.pth"
)
model = models.mobilenet_v3_small()
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)  # Adjust the classifier layer
model.load_state_dict(torch.load(model_path))
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


# Hook for extracting feature maps
class FeatureMapExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        self.target_layer.register_forward_hook(forward_hook)

    def get_feature_maps(self, input_tensor):
        self.model(input_tensor)
        return self.feature_maps


# Plot and save feature maps in a grid
def plot_feature_maps(model, image_path, output_path):
    image = Image.open(image_path)
    image_tensor = preprocess_image(image_path)

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

    num_layers = len(layers)
    grid_size = int(np.ceil(np.sqrt(num_layers + 1)))

    plt.figure(figsize=(20, 20))

    # Plot original image
    plt.subplot(grid_size, grid_size, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image")

    # Plot feature maps for each layer
    for i, layer in enumerate(layers):
        extractor = FeatureMapExtractor(model, layer)
        feature_maps = extractor.get_feature_maps(image_tensor)
        feature_map = feature_maps[0].cpu().detach().numpy()

        # Plot each channel of the feature map
        for j in range(feature_map.shape[0]):
            plt.subplot(grid_size, grid_size, i + 2)
            plt.imshow(feature_map[j])
            plt.axis("off")
            plt.title(f"Layer {i} - Channel {j}")
            break  # Only plot the first channel for simplicity

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    image_path = "./casting_data/casting_data/g_val/ok_front/cast_ok_0_8.jpeg"  # Replace with your image path
    output_path = (
        "feature_maps_grid_output.png"  # Replace with your desired output path
    )
    plot_feature_maps(model, image_path, output_path)
