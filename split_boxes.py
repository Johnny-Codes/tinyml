import os
import shutil
import random


def create_dir_structure(base_dir):
    for split in ["train", "val", "test"]:
        for category in ["intact", "damaged"]:
            os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)


def split_data(source_dir, dest_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    create_dir_structure(dest_dir)

    for category in ["intact", "damaged"]:
        category_dir = os.path.join(source_dir, category)
        images = os.listdir(category_dir)
        random.shuffle(images)

        train_split = int(train_ratio * len(images))
        val_split = int(val_ratio * len(images))

        train_images = images[:train_split]
        val_images = images[train_split : train_split + val_split]
        test_images = images[train_split + val_split :]

        for image in train_images:
            shutil.move(
                os.path.join(category_dir, image),
                os.path.join(dest_dir, "train", category, image),
            )

        for image in val_images:
            shutil.move(
                os.path.join(category_dir, image),
                os.path.join(dest_dir, "val", category, image),
            )

        for image in test_images:
            shutil.move(
                os.path.join(category_dir, image),
                os.path.join(dest_dir, "test", category, image),
            )


if __name__ == "__main__":
    source_directory = "./boxes"
    destination_directory = "./boxes_split"
    split_data(source_directory, destination_directory)
