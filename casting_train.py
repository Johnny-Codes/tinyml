import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchinfo import summary
import matplotlib.pyplot as plt

from helper_functions.image_helper_functions import (
    save_plot,
    save_sample_images,
    visualize_feature_maps,
    visualize_feature_maps_per_layer,
    visualize_feature_maps_black_bg,
)
from helper_functions.general_helper_functions import (
    save_model,
    save_class_labels,
)

from helper_functions.data_helper_functions import (
    save_metrics_to_json,
    save_metrics_to_csv,
)


def visualize_kernels(model, layer_name, num_kernels=6, save_dir="./filter_kernels"):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get the weights of the specified layer
    layer = dict(model.named_modules())[layer_name]
    if isinstance(layer, nn.Conv2d):
        kernels = layer.weight.data.cpu().numpy()
    else:
        raise ValueError(f"Layer {layer_name} is not a Conv2d layer")

    # Plot the kernels
    fig, axes = plt.subplots(1, num_kernels, figsize=(20, 5))
    for i, ax in enumerate(axes):
        if i >= kernels.shape[0]:
            break
        kernel = kernels[i, 0, :, :]
        ax.imshow(kernel, cmap="gray")
        ax.axis("off")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{layer_name}_kernels.png"))
    plt.close()


def deconv_visualization(model, layer_name, input_image, save_dir="./deconv_maps"):
    os.makedirs(save_dir, exist_ok=True)

    # Hook to capture the feature maps
    def hook_fn(module, input, output):
        global feature_maps
        feature_maps = output

    # Register the hook
    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    # Forward pass to get the feature maps
    model.eval()
    with torch.no_grad():
        _ = model(input_image.unsqueeze(0))

    # Remove the hook
    hook.remove()

    # Deconvolution
    deconv = nn.ConvTranspose2d(
        feature_maps.size(1), feature_maps.size(1), kernel_size=3, stride=1, padding=1
    )
    deconv.weight.data = layer.weight.data
    deconv.bias.data = layer.bias.data

    deconv_maps = deconv(feature_maps)

    # Plot the deconvolution maps
    fig, axes = plt.subplots(1, min(6, deconv_maps.size(1)), figsize=(20, 5))
    for i, ax in enumerate(axes):
        if i >= deconv_maps.size(1):
            break
        ax.imshow(deconv_maps[0, i].cpu().numpy(), cmap="gray")
        ax.axis("off")

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{layer_name}_deconv.png"))
    plt.close()


def main():
    # Load boxes dataset
    data_dir = "./casting_data/casting_data/"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "test")

    # Define the data transformations
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

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms["val"])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    classes = ["background", "defective", "non-defective"]
    # Define the model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(classes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    model = model.to(device)

    # Freeze layers up to block 10
    # freeze = True
    # for name, param in model.named_parameters():
    #     if "features.10" in name:
    #         freeze = False
    #     param.requires_grad = not freeze

    # # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # # Unfreeze the parameters of the classification layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    print(
        summary(
            model=model,
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
        )
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 5
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print(f"Epoch {epoch}, Learning Rate: {scheduler.get_last_lr()[0]}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)
        val_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc)

        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        scheduler.step()
        print()

    print("Training complete")

    # Save loss and accuracy plots
    save_plot(train_losses, "Training Loss", "Loss", "Epoch", "train_loss.png")
    save_plot(val_losses, "Validation Loss", "Loss", "Epoch", "val_loss.png")
    save_plot(
        [x.cpu().numpy() for x in train_accuracies],
        "Training Accuracy",
        "Accuracy",
        "Epoch",
        "train_accuracy.png",
    )
    save_plot(
        [x.cpu().numpy() for x in val_accuracies],
        "Validation Accuracy",
        "Accuracy",
        "Epoch",
        "val_accuracy.png",
    )

    # Visualize feature maps
    # inputs, labels = next(iter(val_loader))
    # input_image = inputs[0].to(device)
    # visualize_feature_maps(model, "features.0.0", input_image, device)

    # Visual feature maps of all layers
    # inputs, labels = next(iter(val_loader))
    # input_image = inputs[0].to(device)
    # visualize_feature_maps_per_layer(model, input_image, device)

    # Save the trained model
    # save_model(model, "casting_trained_model.pth")

    # Save the class labels
    # save_class_labels(train_dataset.classes, "casting_class_labels.json")

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": [x.cpu().numpy() for x in train_accuracies],
        "val_accuracies": [x.cpu().numpy() for x in val_accuracies],
    }

    # Save metrics to JSON and CSV
    save_metrics_to_json(metrics)
    save_metrics_to_csv(metrics)

    visualize_kernels(model, "features.0.0")

    inputs, labels = next(iter(val_loader))
    input_image = inputs[0].to(device)
    deconv_visualization(model, "features.0.0", input_image)


if __name__ == "__main__":
    main()
