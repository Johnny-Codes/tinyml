import os
import json
import torch


def save_model(model, model_name, training_mode, epoch, val_acc):
    """
    Saves a PyTorch model to the given path.

    Args:
        model: The PyTorch model to save.
        path: The path to save the model to.
    """
    model_dir = f"./models/{model_name}/{training_mode}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        model.state_dict(), f"{model_dir}/{model_name}-{epoch}-{val_acc:.4f}.pth"
    )


def save_class_labels(class_labels, path):
    """
    Saves class labels to a file in JSON format.

    Args:
        class_labels: A dictionary or list containing class labels to be saved.
        path: The file path where the class labels will be saved.
    """

    with open(path, "w") as f:
        json.dump(class_labels, f)


def save_metrics_to_json(metrics, model_name_with_timestamp, filename="metrics.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[model_name_with_timestamp] = metrics

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
