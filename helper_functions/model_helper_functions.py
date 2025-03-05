import os
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torchinfo import summary
from ultralytics import YOLO


def get_model_name() -> str:
    model_names = {1: "mobilenetv3_small", 2: "mobilenetv3_large", 3: "mobilenetv2"}

    print(
        "Available models:\n"
        "1. mobilenetv3_small\n"
        "2. mobilenetv3_large\n"
        "3. mobilenetv2\n",
        "3. YOLOv8\n",
    )

    model = int(input("Enter the number corresponding to the model: "))

    if model in model_names:
        return model_names[model]
    else:
        raise ValueError(
            "Invalid input. Please enter a number between 1 and 2.",
        )


def define_model(model_name) -> nn.Module:
    if model_name == "mobilenetv3_small":
        from torchvision.models import MobileNet_V3_Small_Weights

        model = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT,
        )
        classification_index = 3
    elif model_name == "mobilenetv3_large":
        from torchvision.models import MobileNet_V3_Large_Weights

        model = models.mobilenet_v3_large(
            weight=MobileNet_V3_Large_Weights.DEFAULT,
        )
        classification_index = 3
    elif model_name == "mobilenetv2":
        from torchvision.models import MobileNet_V2_Weights

        model = models.mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT,
        )
        classification_index = 1
    elif model_name == "YOLOv8":
        handle_yolo_models()
    else:
        raise ValueError("Invalid model name")

    num_features = model.classifier[classification_index].in_features
    model.classifier[classification_index] = nn.Linear(num_features, 2)
    return model


def handle_yolo_models():
    scheme = {}
    model_list = ["yolov8n-cls", "yolov8s-cls", "yolov8x-cls"]
    scheme["model_name"] = int(
        input("Enter the model name:\n1. YOLOv8n\n2. YOLOv8s\n3. YOLOv8x\n")
    )
    scheme["training_type"] = int(
        input("Enter the training type:\n1. fine-tune\n2. full\n")
    )

    scheme["model_name"] = model_list[scheme["model_name"] - 1]

    return 0


def get_training_mode() -> str:
    training_modes = {1: "fine-tune", 2: "full", 3: "partial"}

    print(
        "Available training modes: \n 1. fine-tune\n 2. full\n 3. partial\n",
        "4. quantized aware training\n",
    )

    mode = int(input("Enter the number corresponding to the training mode: "))

    if mode in training_modes:
        return training_modes[mode]
    else:
        raise ValueError(
            "Invalid input. Please enter a number between 1 and 3.",
        )


def print_model_summary(model: nn.Module) -> None:
    print(
        summary(
            model=model,
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
        )
    )


def set_training_mode(model: nn.Module, mode: str, feature_layer=None) -> None:
    """
    feature_layer example: features.10 (10th layer of the model)
    """
    if mode == "fine-tune":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif mode == "partial":
        freeze = True
        for name, param in model.named_parameters():
            if feature_layer in name:
                freeze = False
            param.requires_grad = not freeze
    else:
        raise ValueError("Invalid training mode")


def set_criterion():
    print("\n\nAvailable loss functions:\n" + "1) Cross Entropy Loss\n")
    loss_function = int(input("Which loss function are you using? "))
    if loss_function == 1:
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function")
    return criterion


def set_optimizer(model: nn.Module) -> optim.Optimizer:
    print("\n\nAvailable optimizers: \n\n" + "1) SGD" + "\n2) Adam" + "\n")
    optimizer_name = int(input("Which optimizer are you using? "))
    lr = float(input("What is the initial learning rate? "))
    if optimizer_name == 1:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 2:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer")
    return optimizer


def set_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler:
    print(
        "\n\nAvailable schedulers: \n\n" + "1) StepLR" + "\n2) ReduceLROnPlateau" + "\n"
    )
    scheduler = int(input("Which scheduler are you using? "))
    if scheduler == 1:
        step_size = int(input("What is the step size? (int) "))
        gamma = float(input("What is the gamma? (float) "))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    elif scheduler == 2:
        factor = float(input("What is the factor? (float) (.1 typical): "))
        patience = int(
            input("What is the patience? (int) (default 10 but must enter): "),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
        )
    return scheduler


def get_class_labels_from_directory(directory):
    class_labels = []
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            class_labels.append(subdir)
    return class_labels
