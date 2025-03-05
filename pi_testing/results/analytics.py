import json
import matplotlib.pyplot as plt
import numpy as np

with open("./evaluation_results_test_pi.json", "r") as f:
    data = json.load(f)

interesting_keys = [
    "load_time",
    "ram_usage",
    "file_size",
    "accuracy",
    "average_latency",
    "total_evaluation_time",
    "precision",
    "recall",
    "f1_score",
]

model_names = []
model_data = {}

# Extract data for each model
for key in data.keys():
    if "-16_" in key:
        model_names.append(key)
        model_data[key] = {}
        for i in interesting_keys:
            model_data[key][i] = data[key][i]

# Create bar graphs for each interesting key
for i in interesting_keys:
    values = [model_data[model][i] for model in model_names]
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, values)
    plt.xlabel("Model", fontsize=16)
    plt.ylabel(i, fontsize=16)
    plt.title(f"Comparison of {i} across models", fontsize=20)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{i}_comparison_larger_font.png")
    plt.close()


# # Create 3D scatter plot for latencies, ram_usage, and f1_scores
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection="3d")

# # Extract data for the 3D plot
# latencies = [model_data[model]["average_latency"] for model in model_names]
# ram_usages = [model_data[model]["ram_usage"] for model in model_names]
# f1_scores = [model_data[model]["f1_score"] for model in model_names]

# # Create the scatter plot
# ax.scatter(latencies, ram_usages, f1_scores)

# # Set labels and title
# ax.set_xlabel("Average Latency (s)")
# ax.set_ylabel("RAM Usage")
# ax.set_zlabel("F1 Score")
# ax.set_title("3D Plot of Latency, RAM Usage, and F1 Score")

# # Add model names as annotations
# for i, model in enumerate(model_names):
#     ax.text(latencies[i], ram_usages[i], f1_scores[i], model, zdir="y", ha="left")

# # Save the 3D plot
# plt.savefig("3d_plot.png")
# plt.close()

# Create a plot with two y-axes (Latency and F1 Score)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Extract data for the plot
latencies = [model_data[model]["average_latency"] for model in model_names]
f1_scores = [model_data[model]["f1_score"] for model in model_names]
# ram_usages = [
#     model_data[model]["ram_usage"] for model in model_names
# ]  # Example of a third metric

# Plot latency on the primary y-axis
color = "tab:blue"
ax1.set_xlabel("Model", fontsize=16)
ax1.set_ylabel("Average Latency (s)", color=color, fontsize=16)
ax1.bar(
    model_names,
    latencies,
    color=color,
    width=0.4,
    label="Average Latency (s)",
)
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_xticklabels(model_names, rotation=45, ha="right")  # Rotate x-axis labels

# Create a second y-axis for F1 Score
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

color = "tab:green"
ax2.set_ylabel(
    "F1 Score",
    color=color,
    fontsize=16,
)  # We already handled the x-label with ax1
ax2.plot(model_names, f1_scores, color=color, marker="o", label="F1 Score")
ax2.tick_params(axis="y", labelcolor=color)

# Add a third y-axis for RAM Usage (optional)
# ax3 = ax1.twinx()
# ax3.spines["right"].set_position(("outward", 60))  # Adjust position of the right spine
# color = "tab:green"
# ax3.set_ylabel("RAM Usage (GB)", color=color)
# ax3.plot(
#     model_names, ram_usages, color=color, marker="x", linestyle="--", label="RAM Usage"
# )
# ax3.tick_params(axis="y", labelcolor=color)
# ax3.set_ylim(0, max(ram_usages) * 1.2)  # Adjust y-axis limits for better visibility
# ax3.yaxis.label.set_color(color)  # Set color of the y-axis label

# Ensure that the right-most axis is on top
# ax3.set_zorder(-1)
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# Add title and layout
plt.title("Model Comparison: Latency and F1 Score", fontsize=20)
fig.tight_layout()  # Otherwise the right y-label is slightly clipped

# Add legend
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=12)

# Save the plot
plt.savefig("latency_f1_comparison_larger_font.png", bbox_inches="tight")
plt.close()
