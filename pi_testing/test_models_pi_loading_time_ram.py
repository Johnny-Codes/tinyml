import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, mobilenet_v2
import time
import json
import psutil
import os
from models_and_paths_dir import models_and_paths

models_and_paths = models_and_paths


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
    model = get_model_architecture(model_name)

    # Adjust the classifier layer to match the number of classes
    if model_name.startswith("mobilenet_v3"):
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == "mobilenet_v2":
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
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

    # Unload the model to free up RAM
    del model

# Save results to a JSON file
with open(f"{results_dir}model_loading_times_and_ram.json", "w") as f:
    json.dump(results, f, indent=4)
