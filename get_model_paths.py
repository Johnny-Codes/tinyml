import os


def list_files_in_directory(directory, output_file):
    with open(output_file, "w") as f:
        for root, _, files in os.walk(directory):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                f.write(relative_path + "\n")


if __name__ == "__main__":
    models_directory = "./models"
    output_file = "model_files.txt"
    list_files_in_directory(models_directory, output_file)
    print(f"List of model files saved to {output_file}")
