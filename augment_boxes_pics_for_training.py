import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import shutil

# Define the transformations
transformations = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2,
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    transforms.RandomAffine(degrees=0, shear=10),
]

# Define the original and augmented dataset directories
dataset_dirs = {
    "train_damaged": "./boxes_split/train/damaged/",
    "train_intact": "./boxes_split/train/intact/",
    "val_damaged": "./boxes_split/val/damaged/",
    "val_intact": "./boxes_split/val/intact/",
    "test_damaged": "./boxes_split/test/damaged/",
    "test_intact": "./boxes_split/test/intact/",
}

augmented_dataset_dirs = {
    "train_damaged_augmented": "./boxes_augmented/train/damaged/",
    "train_intact_augmented": "./boxes_augmented/train/intact/",
    "val_damaged_augmented": "./boxes_augmented/val/damaged/",
    "val_intact_augmented": "./boxes_augmented/val/intact/",
    "test_damaged_augmented": "./boxes_augmented/test/damaged/",
    "test_intact_augmented": "./boxes_augmented/test/intact/",
}


# Function to augment dataset
def augment_dataset(original_dir, augmented_dir):
    # Create the augmented dataset directory if it doesn't exist
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
    os.makedirs(augmented_dir)

    # Load the original dataset
    dataset = ImageFolder(original_dir)

    # Apply transformations and save augmented images
    for idx, (img, label) in enumerate(dataset):
        class_dir = os.path.join(augmented_dir, dataset.classes[label])
        os.makedirs(class_dir, exist_ok=True)

        # Save the original image
        img.save(os.path.join(class_dir, f"{idx}_original.jpg"))

        # Apply each transformation and save the augmented images
        for i, transform in enumerate(transformations):
            transformed_img = transform(
                img.convert("RGB")
            )  # Ensure image is in RGB mode
            transformed_img.save(os.path.join(class_dir, f"{idx}_augmented_{i}.jpg"))

    print(f"Dataset augmentation completed for {original_dir}.")


# Augment train, val, and test datasets
for key in dataset_dirs:
    augment_dataset(dataset_dirs[key], augmented_dataset_dirs[key])
