import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, mobilenet_v2
import os
import torch.backends.quantized

models_and_paths = {
    "mnv3s-fine-16": "mobilenetv3_small/fine-tune/mobilenetv3_small-2-t16-v16-0.9958.pth",
    "mnv3s-fine-32": "mobilenetv3_small/fine-tune/mobilenetv3_small-4-t32-v16-0.9972.pth",
    "mnv3s-fine-64": "mobilenetv3_small/fine-tune/mobilenetv3_small-13-t64-v16-0.9972.pth",
    "mnv3s-full-16": "mobilenetv3_small/full/mobilenetv3_small-2-t16-v16-0.9986.pth",
    "mnv3s-full-32": "mobilenetv3_small/full/mobilenetv3_small-1-t32-v16-0.9986.pth",
    "mnv3s-full-64": "mobilenetv3_small/full/mobilenetv3_small-2-t64-v16-0.9986.pth",
    "mnv3l-fine-16": "mobilenetv3_large/fine-tune/mobilenetv3_large-49-t16-v16-0.8294.pth",
    "mnv3l-fine-32": "mobilenetv3_large/fine-tune/mobilenetv3_large-34-t32-v16-0.8210.pth",
    "mnv3l-fine-64": "mobilenetv3_large/fine-tune/mobilenetv3_large-36-t64-v16-0.8252.pth",
    "mnv3l-full-16": "mobilenetv3_large/full/mobilenetv3_large-11-t16-v16-0.9986.pth",
    "mnv3l-full-32": "mobilenetv3_large/full/mobilenetv3_large-13-t32-v16-0.9958.pth",
    "mnv3l-full-64": "mobilenetv3_large/full/mobilenetv3_large-16-t64-v16-0.9958.pth",
    "mnv2-fine-16": "mobilenetv2/fine-tune/mobilenetv2-32-t16-v16-0.9958.pth",
    "mnv2-fine-32": "mobilenetv2/fine-tune/mobilenetv2-4-t32-v16-0.9944.pth",
    "mnv2-fine-64": "mobilenetv2/fine-tune/mobilenetv2-25-t64-v16-0.9944.pth",
    "mnv2-full-16": "mobilenetv2/full/mobilenetv2-2-t16-v16-0.9986.pth",
    "mnv2-full-32": "mobilenetv2/full/mobilenetv2-0-t32-v16-0.9972.pth",
    "mnv2-full-64": "mobilenetv2/full/mobilenetv2-0-t64-v16-0.9986.pth",
}


def get_model_architecture(model_name):
    if model_name == "mobilenet_v3_large":
        return mobilenet_v3_large(weights=None)
    elif model_name == "mobilenet_v3_small":
        return mobilenet_v3_small(weights=None)
    elif model_name == "mobilenet_v2":
        return mobilenet_v2(weights=None)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def apply_dynamic_quantization(model, model_path):
    torch.backends.quantized.engine = "fbgemm"
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    dir_name, file_name = os.path.split(model_path)
    quantized_model_path = os.path.join(dir_name, f"ptdq_{file_name}")
    os.makedirs(dir_name, exist_ok=True)
    torch.save(model_quantized.state_dict(), quantized_model_path)
    print(f"Dynamic quantized model saved as {quantized_model_path}")


def apply_static_quantization(model, model_path, calibration_loader):
    model.eval()
    torch.backends.quantized.engine = "fbgemm"
    print(f"Quantization engine: {torch.backends.quantized.engine}")  # verify engine
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model_prepared = torch.quantization.prepare(model)
    print("Model prepared for quantization.")  # verify prepare step

    # Calibration step (using the prepared model)
    with torch.no_grad():
        print("Starting calibration...")  # verify calibration
        for inputs, _ in calibration_loader:
            inputs = inputs.to("cpu")
            model_prepared(inputs)
        print("Calibration complete.")  # verify calibration completion

    model_quantized = torch.quantization.convert(model_prepared)
    print("Model converted to quantized model.")  # verify convert step

    # Save the *quantized* model (entire model, not just state_dict)
    dir_name, file_name = os.path.split(model_path)
    quantized_model_path = os.path.join(dir_name, f"ptsq_{file_name}")
    os.makedirs(dir_name, exist_ok=True)
    torch.save(model_quantized, quantized_model_path)
    print(f"Static quantized model saved as {quantized_model_path}")


def main():
    data_dir = "./"
    val_dir = os.path.join(data_dir, "./casting_data/casting_data/g_val")

    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=data_transforms,
    )

    validation_batch_size = 32
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=4,
    )

    for k, v in models_and_paths.items():
        main_dir = "./models/casting/"
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

        original_model = get_model_architecture(model_name)

        # Adjust the classifier layer to match the number of classes
        if model_name.startswith("mobilenet_v3"):
            original_model.classifier[3] = torch.nn.Linear(
                original_model.classifier[3].in_features, 2
            )
        elif model_name == "mobilenet_v2":
            original_model.classifier[1] = torch.nn.Linear(
                original_model.classifier[1].in_features, 2
            )

        original_model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        original_model = original_model.to("cpu")

        # Create Dynamic Quantized Model
        # dynamic_model = get_model_architecture(model_name)
        # if model_name.startswith("mobilenet_v3"):
        #     dynamic_model.classifier[3] = torch.nn.Linear(
        #         dynamic_model.classifier[3].in_features, 2
        #     )
        # elif model_name == "mobilenet_v2":
        #     dynamic_model.classifier[1] = torch.nn.Linear(
        #         dynamic_model.classifier[1].in_features, 2
        #     )
        # dynamic_model.load_state_dict(
        #     torch.load(model_path, map_location=torch.device("cpu"))
        # )
        # dynamic_model = dynamic_model.to("cpu")
        # apply_dynamic_quantization(dynamic_model, model_path)

        # Create Static Quantized Model
        static_model = get_model_architecture(model_name)
        if model_name.startswith("mobilenet_v3"):
            static_model.classifier[3] = torch.nn.Linear(
                static_model.classifier[3].in_features, 2
            )
        elif model_name == "mobilenet_v2":
            static_model.classifier[1] = torch.nn.Linear(
                static_model.classifier[1].in_features, 2
            )
        static_model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        static_model = static_model.to("cpu")
        apply_static_quantization(static_model, model_path, val_loader)


if __name__ == "__main__":
    main()
