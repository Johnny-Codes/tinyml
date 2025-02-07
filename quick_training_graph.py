import json
import matplotlib.pyplot as plt

# Load the JSON data
with open("metrics.json", "r") as f:
    data = json.load(f)

# Iterate over each model in the JSON data
for model_name, model_data in data.items():
    training_metrics = model_data["training_metrics"]

    epochs = list(range(training_metrics["total_epochs"]))
    train_acc = [training_metrics[str(epoch)]["train_acc"] for epoch in epochs]
    train_loss = [training_metrics[str(epoch)]["train_loss"] for epoch in epochs]
    val_acc = [training_metrics[str(epoch)]["val_acc"] for epoch in epochs]
    val_loss = [training_metrics[str(epoch)]["val_loss"] for epoch in epochs]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot training accuracy and loss
    ax1.plot(epochs, train_acc, label="Train Accuracy", color="blue")
    ax1.plot(epochs, train_loss, label="Train Loss", color="red")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Value")
    ax1.set_title(f"{model_name} - Training Metrics")
    ax1.legend()

    # Plot validation accuracy and loss
    ax2.plot(epochs, val_acc, label="Validation Accuracy", color="blue")
    ax2.plot(epochs, val_loss, label="Validation Loss", color="red")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Value")
    ax2.set_title(f"{model_name} - Validation Metrics")
    ax2.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{model_name}_training_validation_metrics.png", dpi=600)
    plt.show()
