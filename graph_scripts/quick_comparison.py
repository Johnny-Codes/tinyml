import json
import os
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

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


# Function to extract latencies
def extract_latencies(data):
    return data.get("latencies", [])


# Load data from each file
latency_data = {}
for file_path in file_paths:
    if os.path.exists(file_path):
        data = load_json(file_path)
        model_name = data["model_name"]
        latencies = extract_latencies(data)
        latency_data[model_name] = latencies
    else:
        print(f"File {file_path} does not exist.")

# Perform pairwise t-tests and collect results
results = []
model_names = list(latency_data.keys())
num_models = len(model_names)

for i in range(num_models):
    for j in range(i + 1, num_models):
        model1 = model_names[i]
        model2 = model_names[j]
        latencies1 = latency_data[model1]
        latencies2 = latency_data[model2]

        t_stat, p_value = ttest_ind(latencies1, latencies2)

        results.append(
            {"model1": model1, "model2": model2, "t_stat": t_stat, "p_value": p_value}
        )

# Save the statistical analysis results to a JSON file
with open("latency_statistics.json", "w") as f:
    json.dump(results, f, indent=4)

# Create and save box plots for each model
plt.figure(figsize=(12, 6))
plt.boxplot([latency_data[model] for model in model_names], labels=model_names)
plt.xlabel("Models")
plt.ylabel("Latency (seconds)")
plt.title("Latency Box Plot for Different Models")
plt.xticks(rotation=45)
plt.savefig("latency_box_plots.png", dpi=600)
plt.show()
