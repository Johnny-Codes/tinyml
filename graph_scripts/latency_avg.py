import matplotlib.pyplot as plt
import seaborn as sns

models = ["MobileNetV3 Small", "MobileNetV3 Large", "MobileNetV2", "yolo11n", "yolo11x"]
avg_latency_ms = [0.08562, 0.20919, 0.27512, 0.14830, 2.19082]
avg_latency_secs = [x * 100 for x in avg_latency_ms]  # Convert to milliseconds
colors = ["#7A41BE", "#BE7A41", "#41BE7A", "#254441", "#9FC2CC"]

# Set a visually appealing theme
sns.set_theme(style="whitegrid")

# Create the bar plot with adjusted figure size for better readability
fig, ax = plt.subplots(figsize=(12, 8))  # Increase width to 12 inches
bars = ax.bar(models, avg_latency_secs, color=colors, edgecolor="black", linewidth=1.4)

# Add labels and title with adjusted font sizes
ax.set_xlabel("Model", fontsize=16, labelpad=10)
ax.set_ylabel("Average Latency (ms)", fontsize=16, labelpad=10)
ax.set_title("Average Latency Comparison", fontsize=18, pad=15)

# Customize tick labels
ax.tick_params(axis="x", labelsize=16, rotation=0)
ax.tick_params(axis="y", labelsize=16)

# Add annotations to the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        yval / 2,
        round(yval, 2),  # Display latency value
        ha="center",
        va="center",
        fontsize=14,
        color="white",
    )

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# Save the plot with high DPI
plt.savefig("latency_comparison_with_yolo.png", dpi=600)

# Show the plot
plt.show()
