import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, mobilenet_v2
import time
import json
import psutil
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from models_and_paths_dir import models_and_paths

models_and_paths = models_and_paths
torch.backends.quantized.engine = "qnnpack"


def get_model_architecture(model_name, quantized=False):
    if model_name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=None)
    elif model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if quantized == "ptdq":
        print("================> ptdq")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    print("================> model loaded")
    return model


# Create results directory if it doesn't exist
results_dir = "./results/"
os.makedirs(results_dir, exist_ok=True)

results = []

for k, v in models_and_paths.items():
    main_dir = "../models/casting/"
    model_path = os.path.join(main_dir, v)

    if "mnv3s" in k:
        model_name = "mobilenet_v3_small"
    elif "mnv3l" in k:
        model_name = "mobilenet_v3_large"
    elif "mnv2" in k:
        model_name = "mobilenet_v2"
    else:
        print(f"Unknown model type for {k}")
        continue

    print(f"Loading {k} as {model_name}")

    # Measure loading time
    start_time = time.time()
    if "ptdq" in k:
        print("=============> ptdq")
        model = get_model_architecture(model_name, quantized="ptdq")
    else:
        model = get_model_architecture(model_name)

    # Adjust the classifier layer to match the number of classes
    if model_name.startswith("mobilenet_v3"):
        print("=============> mobilenet_v3")
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == "mobilenet_v2":
        print("=============> mobilenet_v2")
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

    load_time = time.time() - start_time

    # Measure RAM usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024**2)  # Convert to MB

    # Get file size
    file_size = os.path.getsize(model_path) / (1024**2)  # Convert to MB

    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    print(
        f"Model {k} loaded in {load_time:.4f} seconds, using {ram_usage:.2f} MB of RAM, file size {file_size:.2f} MB, number of parameters {num_params}"
    )

    # Store results
    results.append(
        {
            "model_name": k,
            "load_time": load_time,
            "ram_usage": ram_usage,
            "file_size": file_size,
            "num_params": num_params,
        }
    )

    # Check if CUDA is available and move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Define the test dataset and dataloader
    test_dir = "../g_test_casting"
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
    true_labels = []
    predicted_labels = []

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
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

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
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save the latencies and results to a JSON file
    results.append(
        {
            "model_name": k,
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
    )

    with open(f"{results_dir}{k}-results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Unload the model to free up RAM
    del model

# Save combined results to a JSON file
with open(f"{results_dir}combined_results.json", "w") as f:
    json.dump(results, f, indent=4)
