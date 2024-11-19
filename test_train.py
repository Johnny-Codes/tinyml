import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

# Define data transformations
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


# Custom dataset class to handle subdirectories
class CustomImageFolder(datasets.DatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
    ):
        super(CustomImageFolder, self).__init__(
            root,
            loader,
            datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples


def save_plot(data, title, ylabel, xlabel, filename):
    plt.figure()
    sns.set(style="whitegrid")
    plt.plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename)
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
    inputs = inputs.cpu()
    labels = labels.cpu()
    preds = preds.cpu()
    num_images = min(len(inputs), 16)  # Limit to 16 images
    fig = plt.figure(figsize=(15, 15))
    for i in range(num_images):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        img = transforms.ToPILImage()(inputs[i])
        ax.imshow(img)
        ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
    plt.savefig(filename)
    plt.close()


def visualize_feature_maps(model, layer_name, input_image, device):
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
    plt.savefig(f"feature_maps_{layer_name}.png")
    plt.close()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def main():
    # Load the dataset
    data_dir = "./boxes"
    full_dataset = CustomImageFolder(data_dir, transform=data_transforms["train"])

    # Define the model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(full_dataset.classes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 100
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Randomly split the dataset into training and validation sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Update the transforms for the validation dataset
        val_dataset.dataset.transform = data_transforms["val"]

        # Create dataloaders
        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size=32, shuffle=True, num_workers=4
            ),
            "val": DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4),
        }
        dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Save sample images with predictions
        if epoch % 5 == 0:  # Save every 5 epochs
            inputs, labels = next(iter(dataloaders["val"]))
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            save_sample_images(
                inputs,
                labels,
                preds,
                full_dataset.classes,
                f"sample_images_epoch_{epoch}.png",
            )

        print()

    print("Training complete")

    # Save loss and accuracy plots
    save_plot(train_losses, "Training Loss", "Loss", "Epoch", "train_loss.png")
    save_plot(val_losses, "Validation Loss", "Loss", "Epoch", "val_loss.png")
    save_plot(
        train_accuracies, "Training Accuracy", "Accuracy", "Epoch", "train_accuracy.png"
    )
    save_plot(
        val_accuracies, "Validation Accuracy", "Accuracy", "Epoch", "val_accuracy.png"
    )

    # Visualize feature maps
    inputs, labels = next(iter(dataloaders["val"]))
    input_image = inputs[0]
    visualize_feature_maps(model, "features.0.0", input_image, device)

    # Save the trained model
    save_model(model, "trained_model.pth")


if __name__ == "__main__":
    main()
