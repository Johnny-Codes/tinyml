import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
import os
from helper_functions.image_helper_functions import plot_confusion_matrix

# Load the model
model_path = (
    "./models/casting/mobilenetv3_small/fine-tune/mobilenetv3_small-8-0.9876.pth"
)
model = mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)  # Assuming 2 classes
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the test dataset and dataloader without random transformations
test_dir = "./casting_data/casting_data/test_augmented/"
transform_no_aug = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_dataset_no_aug = ImageFolder(test_dir, transform=transform_no_aug)
test_loader_no_aug = DataLoader(test_dataset_no_aug, batch_size=32, shuffle=False)

# Define the test dataset and dataloader with random transformations
transform_with_aug = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_dataset_with_aug = ImageFolder(test_dir, transform=transform_with_aug)
test_loader_with_aug = DataLoader(test_dataset_with_aug, batch_size=32, shuffle=False)


# Function to run the model and collect predictions and true labels
def evaluate_model(test_loader):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())
    return y_true, y_pred


# Evaluate the model on both datasets
y_true_no_aug, y_pred_no_aug = evaluate_model(test_loader_no_aug)
y_true_with_aug, y_pred_with_aug = evaluate_model(test_loader_with_aug)

# Define class names
class_names = test_dataset_no_aug.classes

# Plot and save the confusion matrices
output_path_no_aug = "./confusion_matrix_no_aug.png"
plot_confusion_matrix(y_true_no_aug, y_pred_no_aug, class_names, output_path_no_aug)

output_path_with_aug = "./confusion_matrix_with_aug.png"
plot_confusion_matrix(
    y_true_with_aug, y_pred_with_aug, class_names, output_path_with_aug
)
