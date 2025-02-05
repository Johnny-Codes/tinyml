import json
import matplotlib.pyplot as plt

# Load the metrics data from the JSON file
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Extract the data for the key "mobilenetv3_large-full"
data = metrics["mobilenetv3_large-full"]

# Extract training metrics, filtering out non-numeric keys
epochs = sorted(
    [key for key in data["training_metrics"].keys() if key.isdigit()], key=int
)
train_loss = [data["training_metrics"][epoch]["train_loss"] for epoch in epochs]
train_acc = [data["training_metrics"][epoch]["train_acc"] for epoch in epochs]
val_loss = [data["training_metrics"][epoch]["val_loss"] for epoch in epochs]
val_acc = [data["training_metrics"][epoch]["val_acc"] for epoch in epochs]

# Create the combined graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.plot(epochs, val_acc, label="Validation Accuracy")

# Add labels and title
plt.xlabel("Epochs")
plt.ylabel("Metrics")
plt.title("Training and Validation Metrics")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("combined_metrics_graph.png", dpi=600)
plt.close()
