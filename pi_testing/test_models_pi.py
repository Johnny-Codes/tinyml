import os
import time
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, mobilenet_v2
from models_and_paths_dir import models_and_paths
import psutil
import numpy as np

# Set the quantization engine to qnnpack
torch.backends.quantized.engine = "qnnpack"

# Define the image transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Load the dataset
data_dir = "../g_test_casting"
dataset = ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize statistics
results = {}


def prepare_static_quantized_model(model):
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(
            quant_min=0, quant_max=127
        ),
        weight=torch.quantization.PerChannelMinMaxObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8
        ),
    )
    torch.quantization.prepare(model, inplace=True)
    return model


def get_model_architecture(model_name, quantized=False, model_path=None):
    if model_name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if quantized == "ptdq":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif quantized == "ptsq":
        model = torch.load(model_path)
        print(type(model))
        print(torch.backends.quantized.engine)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.quantized.Conv2d):
                print(f"{name} is quantized")
        print("================> model loaded")

    return model


# Evaluate each model
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, "evaluation_results_test.json")

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

    start_load_time = time.time()
    if "ptsq" in k:
        quantized_model_path = os.path.join(
            main_dir, f"{v}"
        )  # Path to saved quantized model
        model = get_model_architecture(
            model_name, quantized="ptsq", model_path=quantized_model_path
        )
        device = "cpu"
    elif "ptdq" in k:
        model = get_model_architecture(model_name, quantized="ptdq")
        device = "cpu"
        quantized_model_path = os.path.join(main_dir, v)
    else:
        model = get_model_architecture(model_name)
        device = "cpu"
        quantized_model_path = model_path
    load_time = time.time() - start_load_time

    # Load the fine-tuned model weights
    if "ptsq" not in k:
        # Load the fine-tuned model weights (only for non-ptsq models)
        model.load_state_dict(torch.load(quantized_model_path, map_location=device))
        model = model.to(device)  # Move non ptsq models to device

    # Remove this line for ptsq models.
    if "ptsq" not in k:
        model.eval()

    # Get model statistics
    ram_usage = psutil.virtual_memory().used / (1024**3)  # in GB
    file_size = os.path.getsize(quantized_model_path) / (1024**2)  # in MB
    if "ptsq" not in k:
        num_params = sum(p.numel() for p in model.parameters())
    else:
        num_params = "N/A (Quantized Model)"

    results[k] = {
        "load_time": load_time,
        "ram_usage": ram_usage,
        "file_size": file_size,
        "num_params": num_params,
    }

    start_eval_time = time.time()
    latencies = []
    true_labels = []
    predicted_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Measure latency
        start_latency = time.time()
        if "ptsq" in k:
            with torch.no_grad():
                outputs = model(images)
        else:
            outputs = model(images)
        end_latency = time.time()

        latency = end_latency - start_latency
        latencies.append(latency)

        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

    total_eval_time = time.time() - start_eval_time

    # Calculate statistics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(
        true_labels, predicted_labels, output_dict=True, zero_division=1
    )
    num_images = len(true_labels)
    avg_latency = sum(latencies) / num_images

    results[k].update(
        {
            "accuracy": accuracy,
            "average_latency": avg_latency,
            "total_evaluation_time": total_eval_time,
            "num_images": num_images,
            "latencies": [float(lat) for lat in latencies],
            "true_labels": [int(label) for label in true_labels],
            "predicted_labels": [int(pred) for pred in predicted_labels],
            "confusion_matrix": conf_matrix.tolist(),
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": f1,
        }
    )

# Save results to JSON file after all evaluations
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)
