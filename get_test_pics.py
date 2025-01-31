import os
import random
import shutil


def create_test_set(source_dir, dest_dir, num_samples=100):
    classes = ["def_front", "ok_front"]

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        dest_class_dir = os.path.join(dest_dir, class_name)

        if not os.path.exists(dest_class_dir):
            os.makedirs(dest_class_dir)

        images = os.listdir(class_dir)
        selected_images = random.sample(images, num_samples)

        for image in selected_images:
            src_path = os.path.join(class_dir, image)
            dest_path = os.path.join(dest_class_dir, image)
            shutil.copy(src_path, dest_path)

    print(f"Copied {num_samples} images from each class to {dest_dir}")


if __name__ == "__main__":
    source_directory = "./casting_data/casting_data/train"
    destination_directory = "./casting_data/casting_data/test"
    create_test_set(source_directory, destination_directory)
