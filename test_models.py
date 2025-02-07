import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
import os
from helper_functions.image_helper_functions import plot_confusion_matrix

import torchvision.transforms as transforms

# Load the pretrained model
model_path = (
    "./models/casting/mobilenetv3_small/fine-tune/mobilenetv3_small-4-0.9972.pth"
)
model = mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)  # Assuming binary classification
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the dataset and dataloader
data_dir = "./casting_data/casting_data/test"
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
dataset = ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model and collect predictions
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Plot and save the confusion matrix
class_names = dataset.classes
output_path = "./confusion_matrix.png"
plot_confusion_matrix(y_true, y_pred, class_names, output_path)
