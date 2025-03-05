import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import (
    mobilenet_v3_large,
    mobilenet_v3_small,
    mobilenet_v2,
)
import time
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import os
from pi_testing.models_and_paths_dir import models_and_paths


mv2_models = models_and_paths


def get_model_architecture(model_name):
    if model_name == "mobilenet_v3_large":
        return mobilenet_v3_large(weights=None)
    elif model_name == "mobilenet_v3_small":
        return mobilenet_v3_small(weights=None)
    elif model_name == "mobilenet_v2":
        return mobilenet_v2(weights=None)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# Create results directory if it doesn't exist
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)

for key, model_path in mv2_models.items():
    # Load the model architecture
    model_name = "mobilenet_v3_small"
    model_architecture = get_model_architecture(model_name)

    # Load the model weights
    model_path = f"../models/casting/{model_path}"
    model = model_architecture
    if model_name.startswith("mobilenet_v3"):
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == "mobilenet_v2":
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

    # Check if CUDA is available and move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define the test dataset and dataloader
    test_dir = "./g_test_casting"
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )  # Batch size of 1 for latency measurement

    # Measure latency for each image classification
    latencies = []
    y_true = []
    y_pred = []
    num_images = 0

    # Start total evaluation timer
    total_start_time = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)

            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

            num_images += 1

    # End total evaluation timer
    total_end_time = time.time()
    total_evaluation_time = total_end_time - total_start_time

    # Calculate average latency
    average_latency = sum(latencies) / len(latencies)
    print(f"Average latency per image: {average_latency:.6f} seconds")
    print(f"Total evaluation time: {total_evaluation_time:.6f} seconds")
    print(f"Number of images tested: {num_images}")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()  # Convert to list for JSON serialization

    # Compute precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save the latencies and results to a JSON file
    results = {
        "model_name": key,
        "average_latency": average_latency,
        "total_evaluation_time": total_evaluation_time,
        "num_images": num_images,
        "latencies": latencies,
        "true_labels": y_true,
        "predicted_labels": y_pred,
        "confusion_matrix": cm_list,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    with open(f"{results_dir}{key}-latencies.json", "w") as f:
        json.dump(results, f, indent=4)

    # Create and save a box plot of the latencies
    plt.figure(figsize=(10, 6))
    plt.boxplot(latencies)
    plt.title("Latency Box Plot")
    plt.ylabel("Latency (seconds)")
    plt.savefig(f"{results_dir}{key}-latency-boxplot.png")
    plt.close()
