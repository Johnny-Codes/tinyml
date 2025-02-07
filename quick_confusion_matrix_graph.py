import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file paths
file_paths = [
    "mnv3s-16-latencies.json",
    "mnv3s-32-latencies.json",
    "mnv3s-64-latencies.json",
    "mnv3s-full-16-latencies.json",
    "mnv3s-full-32-latencies.json",
    "mnv3s-full-64-latencies.json",
    "mnv3l-fine-16-latencies.json",
    "mnv3l-fine-32-latencies.json",
    "mnv3l-fine-64-latencies.json",
    "mnv3l-full-16-latencies.json",
    "mnv3l-full-32-latencies.json",
    "mnv3l-full-64-latencies.json",
    "mnv2-fine-16-latencies.json",
    "mnv2-fine-32-latencies.json",
    "mnv2-fine-64-latencies.json",
    "mnv2-full-16-latencies.json",
    "mnv2-full-32-latencies.json",
    "mnv2-full-64-latencies.json",
]


# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Function to extract confusion matrix
def extract_confusion_matrix(data):
    return data.get("confusion_matrix", [])


# Iterate over each file and extract confusion matrix
for file_path in file_paths:
    if os.path.exists(file_path):
        data = load_json(file_path)
        confusion_matrix = extract_confusion_matrix(data)

        # Plot confusion matrix as heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {data['model_name']}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig(f"{data['model_name']}_confusion_matrix.png", dpi=600)
        plt.show()
    else:
        print(f"File {file_path} does not exist.")
