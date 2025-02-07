import json
import matplotlib.pyplot as plt

# Define the file paths
file_paths = [
    "mnv3s-fine-16-latencies.json",
    "mnv3s-fine-32-latencies.json",
    "mnv3s-fine-64-latencies.json",
    "mnv3s-full-16-latencies.json",
    "mnv3s-full-32-latencies.json",
    "mnv3s-full-64-latencies.json",
    "mnv3l-fine-16-latencies.json",
    "mnv3l-fine-32-latencies.json",
    "mnv3l-fine-64-latencies.json",
    "mnv3l-full-16-latencies.json",
    "mnv3l-full-32-latencies.json",
    "mnv3l-full-64-latencies.json",
    "mnv2-fine-16-latencies.json",
    "mnv2-fine-32-latencies.json",
    "mnv2-fine-64-latencies.json",
    "mnv2-full-16-latencies.json",
    "mnv2-full-32-latencies.json",
    "mnv2-full-64-latencies.json",
]


# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Load data from each file
latency_data = {}
accuracy_data = {}
average_latency_data = {}
for file_path in file_paths:
    data = load_json(file_path)
    model_name = data["model_name"]
    latencies = data["latencies"]
    true_labels = data["true_labels"]
    predicted_labels = data["predicted_labels"]

    # Calculate accuracy
    correct_predictions = sum(t == p for t, p in zip(true_labels, predicted_labels))
    accuracy = correct_predictions / len(true_labels)

    # Calculate average latency
    average_latency = sum(latencies) / len(latencies)

    latency_data[model_name] = latencies
    accuracy_data[model_name] = accuracy
    average_latency_data[model_name] = average_latency

# Plot average latency vs accuracy
plt.figure(figsize=(12, 6))
colors = {
    "mnv3s-fine": "blue",
    "mnv3s-full": "cyan",
    "mnv3l-fine": "green",
    "mnv3l-full": "lime",
    "mnv2-fine": "red",
    "mnv2-full": "orange",
}
for model_name in average_latency_data.keys():
    prefix = "-".join(model_name.split("-")[:2])
    plt.scatter(
        average_latency_data[model_name],
        accuracy_data[model_name] * 100,
        label=prefix,
        color=colors[prefix],
        edgecolors="black",
        alpha=0.5,
        linewidths=1,
    )

# Create custom legend
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        markeredgecolor="black",
        alpha=0.5,
        markersize=10,
        label=prefix,
    )
    for prefix, color in colors.items()
]
plt.legend(handles=handles, title="Model Groups")

plt.xlabel("Average Latency (seconds)")
plt.ylabel("Accuracy (%)")
plt.title("Average Latency vs Accuracy for Different Models")
plt.savefig("average_latency_vs_accuracy.png", dpi=600)
plt.show()
