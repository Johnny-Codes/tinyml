import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from image_helper_functions import (
    save_plot,
    save_sample_images,
    visualize_feature_maps,
    visualize_feature_maps_per_layer,
)
from general_helper_functions import (
    save_model,
    save_class_labels,
)

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
        phase_transforms=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
    ):
        super(CustomImageFolder, self).__init__(
            root,
            loader,
            datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None,
            transform=None,  # We will set the transform dynamically
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.phase_transforms = phase_transforms

    def set_transform(self, phase):
        self.transform = self.phase_transforms[phase]


def main():
    # Load boxes dataset
    data_dir = "./boxes"
    full_dataset = CustomImageFolder(data_dir, phase_transforms=data_transforms)

    # Define the model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(full_dataset.classes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")
    print(f"using device {device}")
    model = model.to(device)

    """
    need to add preprocessing like in "a real time automated defect detection
    ceramics paper.
    """

    # Freeze layers up to block 10
    # freeze = True
    # for name, param in model.named_parameters():
    #     if "features.10" in name:
    #         freeze = False
    #     param.requires_grad = not freeze

    # # Freeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # # Unfreeze the parameters of the classification layer
    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    # print(
    #     summary(
    #         model=model,
    #         input_size=(
    #             32,
    #             3,
    #             224,
    #             224,
    #         ),  # make sure this is "input_size", not "input_shape"
    #         # col_names=["input_size"], # uncomment for smaller output
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"],
    #     )
    # )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    num_epochs = 10
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print(f"Epoch {epoch}, Learning Rate: {scheduler.get_last_lr()[0]}")
        print("-" * 10)

        # Randomly split the dataset into training and validation sets
        # train_size = int(0.8 * len(full_dataset))
        # val_size = len(full_dataset) - train_size
        # train_dataset, val_dataset = random_split(
        #     full_dataset,
        #     [train_size, val_size],
        # )

        # Create dataloaders
        dataloaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=4,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=4,
            ),
        }
        dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

        for phase in ["train", "val"]:
            full_dataset.set_transform(phase)

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

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
        # if epoch % 5 == 0 or epoch == num_epochs - 1:
        #     inputs, labels = next(iter(dataloaders["val"]))
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)
        #     with torch.no_grad():
        #         outputs = model(inputs)
        #         _, preds = torch.max(outputs, 1)
        #     save_sample_images(
        #         inputs,
        #         labels,
        #         preds,
        #         full_dataset.classes,
        #         f"sample_images_epoch_{epoch}.png",
        #     )
        scheduler.step()
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
    # inputs, labels = next(iter(dataloaders["val"]))
    # input_image = inputs[0]
    # visualize_feature_maps(model, "features.0.0", input_image, device)

    # Visual feature maps of all layers
    # inputs, labels = next(iter(dataloaders["val"]))
    # input_image = inputs[0]
    # visualize_feature_maps_per_layer(model, input_image, device)

    # Save the trained model
    # save_model(model, "trained_model.pth")

    # Save the class labels
    # save_class_labels(full_dataset.classes, "class_labels.json")


if __name__ == "__main__":
    main()
