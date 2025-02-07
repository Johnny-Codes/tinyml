import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import (
    # mobilenet_v3_large,
    # mobilenet_v3_small,
    mobilenet_v2,
)
import time
import json
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

models = {
    # "mnv2-fine-16": "fine-tune/mobilenetv2-9-t16-v16-0.9741.pth",
    # "mnv2-fine-32": "fine-tune/mobilenetv2-6-t32-v16-0.9719.pth",
    # "mnv2-fine-64": "fine-tune/mobilenetv2-9-t64-v16-0.9712.pth",
    "mnv2-full-16": "full/mobilenetv2-7-t16-v16-0.9995.pth",
    "mnv2-full-32": "full/mobilenetv2-5-t32-v16-0.9995.pth",
    "mnv2-full-64": "full/mobilenetv2-2-t64-v16-0.9995.pth",
}

# model_architecture = mobilenet_v3_large(weights=None)
model_architecture = mobilenet_v2(weights=None)
for key in models.keys():
    # Load the model
    model_path = f"./models/casting/mobilenetv2/{models[key]}"
    model = model_architecture
    # for v3
    # model.classifier[3] = torch.nn.Linear(
    #     model.classifier[3].in_features, 2
    # )  # Assuming 2 classes
    # for v2
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    # Check if CUDA is available and move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define the test dataset and dataloader
    test_dir = "./casting_data/casting_data/test_augmented/"
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
    }

    with open(f"{key}-latencies.json", "w") as f:
        json.dump(results, f, indent=4)

    # Create and save a box plot of the latencies
    plt.figure(figsize=(10, 6))
    plt.boxplot(latencies)
    plt.title("Latency Box Plot")
    plt.ylabel("Latency (seconds)")
    plt.savefig(f"{key}-latency-boxplot.png")
    plt.close()
