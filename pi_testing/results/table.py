import json
import matplotlib.pyplot as plt
import numpy as np

with open("./evaluation_results_test_pi.json", "r") as f:
    data = json.load(f)

# Extract data for the table
table_data = {}
for key, value in data.items():
    model_type = None
    training_batch_size = None
    quantization = None

    if "mnv2" in key:
        model_type = "MobileNetV2"
    elif "mnv3l" in key:
        model_type = "MobileNetV3 Large"
    elif "mnv3s" in key:
        model_type = "MobileNetV3 Small"
    else:
        continue

    if "16" in key:
        training_batch_size = 16
    elif "32" in key:
        training_batch_size = 32
    elif "64" in key:
        training_batch_size = 64
    else:
        continue

    if "_nonq" in key:
        quantization = "Non-Quantized"
    elif "_ptdq" in key:
        quantization = "PTDQ"
    elif "_ptsq" in key:
        quantization = "PTSQ"
    else:
        continue

    if model_type not in table_data:
        table_data[model_type] = {}
    if quantization not in table_data[model_type]:
        table_data[model_type][quantization] = {}
    if training_batch_size not in table_data[model_type][quantization]:
        table_data[model_type][quantization][training_batch_size] = {}

    table_data[model_type][quantization][training_batch_size]["Accuracy"] = value.get(
        "accuracy", "N/A"
    )
    table_data[model_type][quantization][training_batch_size]["Latency"] = value.get(
        "average_latency", "N/A"
    )

# Prepare data for the table
model_types = ["MobileNetV2", "MobileNetV3 Large", "MobileNetV3 Small"]
quantization_types = ["Non-Quantized", "PTDQ"]  # Removed "PTSQ"
batch_sizes = [16, 32, 64]
cell_text = []
row_labels = []

for model in model_types:
    for quantization in quantization_types:
        row = []
        row_labels.append(f"{model} ({quantization})")
        for batch_size in batch_sizes:
            if (
                model in table_data
                and quantization in table_data[model]
                and batch_size in table_data[model][quantization]
            ):
                accuracy = table_data[model][quantization][batch_size].get(
                    "Accuracy", "N/A"
                )
                latency = table_data[model][quantization][batch_size].get(
                    "Latency", "N/A"
                )
                if accuracy != "N/A":
                    accuracy_str = f"{accuracy * 100:.2f}%"
                else:
                    accuracy_str = "N/A"
                if latency != "N/A":
                    latency_str = f"{latency * 1000:.2f} ms"
                else:
                    latency_str = "N/A"
                row.append(f"{accuracy_str} @ {latency_str}")
            else:
                row.append("N/A")
        cell_text.append(row)

# Create the table
fig, ax = plt.subplots(figsize=(12, 6))  # Adjusted height for fewer rows
ax.axis("off")

table = plt.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=[str(bs) for bs in batch_sizes],
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8)  # Reduced font size to fit more rows
table.scale(1.2, 1.2)

# Add title
plt.title("Accuracy @ Latency Comparison by Model and Batch Size")

# Save the table
plt.savefig("accuracy_latency_table.png", bbox_inches="tight", dpi=600)
plt.close()
