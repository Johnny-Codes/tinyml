import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

data = {
    "MNV2-Fine-16-PTDQ": {"accuracy": 0.9972, "latency": 0.2757},
    "MNV2-Fine-16-Original": {"accuracy": 0.9972, "latency": 0.27554},
    "MNV3S-Fine-32-Original": {"accuracy": 0.9972, "latency": 0.0866},
    "MNV3S-Fine-32-PTDQ": {"accuracy": 0.9972, "latency": 0.08663},
    "MNV2-Fine-32-Original": {"accuracy": 0.9958, "latency": 0.27582},
    "MNV2-Fine-32-PTDQ": {"accuracy": 0.9958, "latency": 0.27507},
    "MNV2-Fine-64-Original": {"accuracy": 0.9958, "latency": 0.27578},
    "MNV2-Fine-64-PTDQ": {"accuracy": 0.9958, "latency": 0.27513},
    "MNV3-Transfer-64-Original": {
        "accuracy": 0.9958,
        "latency": 0.08581,
    },
    "MNV3S-Fine-64-Original": {"accuracy": 0.9958, "latency": 0.08722},
    "MNV3S-Fine-64-PTDQ": {"accuracy": 0.9958, "latency": 0.08672},
}

colors = ["#7A41BE", "#BE7A41", "#41BE7A"]

# Extract data for plotting
model_names = list(data.keys())
accuracies = [data[model]["accuracy"] for model in model_names]
latencies = [data[model]["latency"] for model in model_names]

# Determine bar colors based on model name
bar_colors = [colors[0] if "MNV2" in name else colors[2] for name in model_names]

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot accuracy as a line graph
ax1.plot(model_names, accuracies, marker="o", color=colors[1], label="Accuracy")
ax1.set_xlabel("Model", fontsize=14)
ax1.set_ylabel("Accuracy", color=colors[1], fontsize=14)
ax1.tick_params(axis="y", labelcolor=colors[1])
ax1.set_ylim(0.995, 0.998)  # Set the y-axis limit for accuracy

# Create a second y-axis for latency
ax2 = ax1.twinx()
ax2.bar(model_names, latencies, color=bar_colors, alpha=0.7)
ax2.set_ylabel("Latency (seconds)", color=colors[0], fontsize=14)
ax2.tick_params(axis="y", labelcolor=colors[0])
# ax2.set_ylim(0, 0.1)  # Set the y-axis limit for latency

# Rotate x-axis labels for better readability
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45, ha="right")
plt.xticks(rotation=45, ha="right")

# Add title and legend
plt.title("Top 11 Model Accuracy and Latency", fontsize=18)
fig.tight_layout(rect=[0, 0.15, 1, 0.95])  # Adjust layout to fit rotated labels

# Create custom legend handles for model types
mnv2_patch = mpatches.Patch(color=colors[0], alpha=0.7, label="MobileNetV2 Latency")
mnv3s_patch = mpatches.Patch(
    color=colors[2], alpha=0.7, label="MobileNetV3 Small Latency"
)

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Add custom handles for model types
ax1.legend(
    lines + lines2 + [mnv2_patch, mnv3s_patch],
    labels + labels2 + ["MobileNetV2 Latency", "MobileNetV3 Small Latency"],
    loc="upper right",
    fontsize=12,
)

# Save the plot
plt.savefig("accuracy_latency_plot.png", dpi=600)

# Show the plot
plt.show()
