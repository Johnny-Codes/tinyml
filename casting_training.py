import os
import datetime
import torch

# import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from helper_functions.image_helper_functions import (
    visualize_feature_maps_per_layer,
)


# from torch.optim.lr_scheduler import StepLR


# import matplotlib.pyplot as plt
# from helper_functions.image_helper_functions import (
#     save_plot,
#     save_sample_images,
#     visualize_feature_maps,
#     visualize_feature_maps_per_layer,
#     visualize_feature_maps_black_bg,
# )
from helper_functions.general_helper_functions import (
    save_metrics_to_json,
)

from helper_functions.data_transforms import mobile_net_data_transforms
from helper_functions.model_helper_functions import (
    define_model,
    print_model_summary,
    set_training_mode,
    get_model_name,
    get_training_mode,
    set_criterion,
    set_optimizer,
    set_scheduler,
    get_class_labels_from_directory,
)
from helper_functions.training_helper_functions import train_model
from helper_functions.image_dataset_helper_functions import (
    get_data_set,
    get_data_set_dirs,
)


def main():

    data_set = get_data_set()
    data_dir = get_data_set_dirs(data_set)

    model_name = get_model_name()

    train_dir = os.path.join(data_dir, "train_augmented")
    val_dir = os.path.join(data_dir, "val_augmented")

    data_transforms = mobile_net_data_transforms

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=data_transforms["train"],
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=data_transforms["val"],
    )

    training_batch_size = int(input("Enter the training batch size: "))
    validation_batch_size = int(input("Enter the validation batch size: "))

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = define_model(model_name)
    model.class_labels = get_class_labels_from_directory(train_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n\nusing device {device} ({torch.cuda.get_device_name(0)})\n\n")
    model = model.to(device)
    training_mode = get_training_mode()

    if training_mode == "partial":
        feature_layer_frozen = input("Which layer are you freezing? i.e. features.10")
        set_training_mode(model, training_mode, feature_layer=feature_layer_frozen)
    else:
        feature_layer_frozen = None
        set_training_mode(model, training_mode)

    criterion = set_criterion()
    optimizer = set_optimizer(model)
    use_scheduler = input("Use scheduler? (y/n) ")
    scheduler = set_scheduler(optimizer) if use_scheduler == "y" else None

    print_summary = input("Print model summary? (y/n) ")
    if print_summary == "y":
        print_model_summary(model)

    training_metrics = train_model(
        model,
        model_name,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        criterion,
        training_mode,
        dataset=data_set,
    )

    model_name_key = f"{model_name}-{training_mode}"

    data = {
        "model_name": model_name,
        "dtg": datetime.datetime.now().strftime("%Y%m%d-%H%M"),
        "training_batch_size": training_batch_size,
        "validation_batch_size": validation_batch_size,
        "device": torch.cuda.get_device_name(0),
        "training_mode": training_mode,
        "feature_layer_frozen": feature_layer_frozen,
        "criterion": criterion.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__ if scheduler else None,
        "training_metrics": training_metrics,
    }

    save_metrics_to_json(data, model_name_key)


if __name__ == "__main__":
    main()
