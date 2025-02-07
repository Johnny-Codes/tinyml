import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the latency data from the JSON files
files = [
    "mnv3s-16-latencies.json",
    "mnv3s-32-latencies.json",
    "mnv3s-64-latencies.json",
]
latencies_data = {}

for file in files:
    with open(file, "r") as f:
        data = json.load(f)
        latencies_data[data["model_name"]] = data["latencies"]

# Perform statistical analysis on each model's latency
stats_summary = {}
for model_name, latencies in latencies_data.items():
    latencies_np = np.array(latencies)
    stats_summary[model_name] = {
        "mean": np.mean(latencies_np),
        "median": np.median(latencies_np),
        "std_dev": np.std(latencies_np),
        "min": np.min(latencies_np),
        "max": np.max(latencies_np),
        "25th_percentile": np.percentile(latencies_np, 25),
        "75th_percentile": np.percentile(latencies_np, 75),
    }

# Perform statistical tests to compare the latencies between models
model_names = list(latencies_data.keys())
comparisons = []

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model1 = model_names[i]
        model2 = model_names[j]
        latencies1 = latencies_data[model1]
        latencies2 = latencies_data[model2]

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(latencies1, latencies2)
        comparisons.append(
            {
                "model1": model1,
                "model2": model2,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": bool(
                    p_value < 0.05
                ),  # Convert to standard Python boolean
            }
        )

# Save the statistical summary and comparison results to a JSON file
results = {"stats_summary": stats_summary, "comparisons": comparisons}

with open("latency_comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Plot box plots for visual comparison
plt.figure(figsize=(12, 8))
plt.boxplot([latencies_data[model] for model in model_names], tick_labels=model_names)
plt.title("Latency Comparison Between Models")
plt.ylabel("Latency (seconds)")
plt.savefig("latency_comparison_boxplot.png")
plt.close()
