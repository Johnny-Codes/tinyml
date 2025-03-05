import os
import json
import torch
import random
import numpy as np


def save_model(
    model,
    model_name,
    dataset,
    training_mode,
    epoch,
    val_acc,
    t_size,
    v_size,
):
    """
    Saves a PyTorch model to the given path.

    Args:
        model: The PyTorch model to save.
        path: The path to save the model to.
    """
    model_dir = f"./models/{dataset}/{model_name}/{training_mode}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        model.state_dict(),
        f"{model_dir}/{model_name}-{epoch}-t{t_size}-v{v_size}-{val_acc:.4f}.pth",
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


def save_metrics_to_json(metrics, model_name_key, filename="metrics.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[model_name_key] = metrics

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Metrics saved to {filename}")


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
