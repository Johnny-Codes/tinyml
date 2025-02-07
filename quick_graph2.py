import json
import matplotlib.pyplot as plt

# Load the JSON data
with open("metrics.json") as f:
    data = json.load(f)

# Initialize lists to store metrics
models = []
val_accs = []
val_losses = []
train_accs = []
train_losses = []

# Extract metrics for each model
for model_name, metrics in data.items():
    models.append(model_name)
    val_acc = [
        metrics["training_metrics"][str(epoch)]["val_acc"]
        for epoch in range(metrics["training_metrics"]["total_epochs"])
    ]
    val_loss = [
        metrics["training_metrics"][str(epoch)]["val_loss"]
        for epoch in range(metrics["training_metrics"]["total_epochs"])
    ]
    train_acc = [
        metrics["training_metrics"][str(epoch)]["train_acc"]
        for epoch in range(metrics["training_metrics"]["total_epochs"])
    ]
    train_loss = [
        metrics["training_metrics"][str(epoch)]["train_loss"]
        for epoch in range(metrics["training_metrics"]["total_epochs"])
    ]

    val_accs.append(val_acc)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    train_losses.append(train_loss)

# Plot validation accuracy and validation loss
plt.figure(figsize=(12, 6))
for i, model in enumerate(models):
    plt.plot(val_accs[i], label=f"{model} val_acc")
    plt.plot(val_losses[i], label=f"{model} val_loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Validation Accuracy and Loss")
plt.legend()
plt.savefig("validation_accuracy_loss.png", dpi=600)
plt.show()

# Plot training accuracy and training loss
plt.figure(figsize=(12, 6))
for i, model in enumerate(models):
    plt.plot(train_accs[i], label=f"{model} train_acc")
    plt.plot(train_losses[i], label=f"{model} train_loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Accuracy and Loss")
plt.legend()
plt.savefig("training_accuracy_loss.png", dpi=600)
plt.show()
