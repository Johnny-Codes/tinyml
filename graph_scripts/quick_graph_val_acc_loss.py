import json
import matplotlib.pyplot as plt
import re

# Load the JSON data
with open("metrics.json") as f:
    data = json.load(f)

# Initialize dictionaries to store metrics for each model type and training mode
metrics_dict = {
    "mobilenetv3_small-fine-tune": {"models": [], "val_accs": [], "val_losses": []},
    "mobilenetv3_small-full": {"models": [], "val_accs": [], "val_losses": []},
    "mobilenetv3_large-fine-tune": {"models": [], "val_accs": [], "val_losses": []},
    "mobilenetv3_large-full": {"models": [], "val_accs": [], "val_losses": []},
    "mobilenetv2-fine-tune": {"models": [], "val_accs": [], "val_losses": []},
    "mobilenetv2-full": {"models": [], "val_accs": [], "val_losses": []},
}

# Extract metrics for each model
for model_name, metrics in data.items():
    if "mobilenetv3_small" in model_name:
        if "fine-tune" in model_name:
            key = "mobilenetv3_small-fine-tune"
        else:
            key = "mobilenetv3_small-full"
    elif "mobilenetv3_large" in model_name:
        if "fine-tune" in model_name:
            key = "mobilenetv3_large-fine-tune"
        else:
            key = "mobilenetv3_large-full"
    elif "mobilenetv2" in model_name:
        if "fine-tune" in model_name:
            key = "mobilenetv2-fine-tune"
        else:
            key = "mobilenetv2-full"
    else:
        continue

    metrics_dict[key]["models"].append(model_name)
    val_acc = []
    val_loss = []
    total_epochs = metrics["training_metrics"]["total_epochs"]

    for epoch in range(total_epochs):
        if str(epoch) in metrics["training_metrics"]:
            val_acc.append(metrics["training_metrics"][str(epoch)]["val_acc"])
            loss = metrics["training_metrics"][str(epoch)]["val_loss"]
            if loss > 1.0:
                loss = 1.0
            val_loss.append(loss)
        else:
            break

    metrics_dict[key]["val_accs"].append(val_acc)
    metrics_dict[key]["val_losses"].append(val_loss)


# Function to plot validation accuracy and validation loss
def plot_metrics(models, val_accs, val_losses, title, output_path):
    plt.figure(figsize=(12, 6))

    # Plot validation accuracy and validation loss on the same subplot
    for i, model in enumerate(models):
        # Extract training batch size from model name using regex
        match = re.search(r"-t_(\d+)-", model)
        if match:
            batch_size = match.group(1)
        else:
            batch_size = "Unknown"  # Handle cases where batch size is not in the name

        plt.plot(val_accs[i], label=f"Training Batch Size {batch_size} Accuracy")
        plt.plot(val_losses[i], label=f"Training Batch Size {batch_size} Loss")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    plt.title(f"{title}: Validation Accuracy and Loss", fontsize=20)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()


# Plot metrics for each model type and training mode
plot_metrics(
    metrics_dict["mobilenetv3_small-fine-tune"]["models"],
    metrics_dict["mobilenetv3_small-fine-tune"]["val_accs"],
    metrics_dict["mobilenetv3_small-fine-tune"]["val_losses"],
    "MobileNetV3 Small Fine-Tune Models",
    "mobilenetv3_small_fine_tune_validation_accuracy_loss_larger_font.png",
)

plot_metrics(
    metrics_dict["mobilenetv3_small-full"]["models"],
    metrics_dict["mobilenetv3_small-full"]["val_accs"],
    metrics_dict["mobilenetv3_small-full"]["val_losses"],
    "MobileNetV3 Small Full Models",
    "mobilenetv3_small_full_validation_accuracy_loss_larger_font.png",
)

plot_metrics(
    metrics_dict["mobilenetv3_large-fine-tune"]["models"],
    metrics_dict["mobilenetv3_large-fine-tune"]["val_accs"],
    metrics_dict["mobilenetv3_large-fine-tune"]["val_losses"],
    "MobileNetV3 Large Fine-Tune Models",
    "mobilenetv3_large_fine_tune_validation_accuracy_loss_larger_font.png",
)

plot_metrics(
    metrics_dict["mobilenetv3_large-full"]["models"],
    metrics_dict["mobilenetv3_large-full"]["val_accs"],
    metrics_dict["mobilenetv3_large-full"]["val_losses"],
    "MobileNetV3 Large Full Models",
    "mobilenetv3_large_full_validation_accuracy_loss_larger_font.png",
)

plot_metrics(
    metrics_dict["mobilenetv2-fine-tune"]["models"],
    metrics_dict["mobilenetv2-fine-tune"]["val_accs"],
    metrics_dict["mobilenetv2-fine-tune"]["val_losses"],
    "MobileNetV2 Fine-Tune Models",
    "mobilenetv2_fine_tune_validation_accuracy_loss_larger_font.png",
)

plot_metrics(
    metrics_dict["mobilenetv2-full"]["models"],
    metrics_dict["mobilenetv2-full"]["val_accs"],
    metrics_dict["mobilenetv2-full"]["val_losses"],
    "MobileNetV2 Full Models",
    "mobilenetv2_full_validation_accuracy_loss_larger_font.png",
)
