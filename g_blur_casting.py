import cv2
import os
import glob


def apply_gaussian_blur(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_path in glob.glob(os.path.join(input_dir, "*.jpeg")):
        img = cv2.imread(img_path)
        if img is not None:
            blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, blurred_img)


def process_directories():
    base_input_dir = "./casting_data/casting_data"
    base_output_dir = "./casting_data/casting_data"
    categories = ["var"]
    subcategories = ["def_front", "ok_front"]

    for category in categories:
        for subcategory in subcategories:
            input_dir = os.path.join(base_input_dir, category, subcategory)
            output_dir = os.path.join(base_output_dir, f"g_{category}", subcategory)
            apply_gaussian_blur(input_dir, output_dir)


if __name__ == "__main__":
    process_directories()
