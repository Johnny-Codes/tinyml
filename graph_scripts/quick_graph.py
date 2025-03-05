import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON data
with open("latency_statistics.json", "r") as f:
    data = json.load(f)

# Extract data for plotting
model1 = [entry["model1"] for entry in data]
model2 = [entry["model2"] for entry in data]
t_stat = [entry["t_stat"] for entry in data]
p_value = [entry["p_value"] for entry in data]

# Create a DataFrame for easier plotting
import pandas as pd

df = pd.DataFrame(
    {"Model 1": model1, "Model 2": model2, "t_stat": t_stat, "p_value": p_value}
)

# Plot t-statistics
plt.figure(figsize=(14, 7))
sns.scatterplot(
    data=df,
    x="Model 1",
    y="t_stat",
    hue="Model 2",
    palette="tab10",
    s=100,
    edgecolor="black",
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Model 1")
plt.ylabel("t-statistic")
plt.title("t-statistics for Model Comparisons")
plt.legend(title="Model 2", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("t_statistics_plot.png", dpi=600)
plt.show()

# Plot p-values
plt.figure(figsize=(14, 7))
sns.scatterplot(
    data=df,
    x="Model 1",
    y="p_value",
    hue="Model 2",
    palette="tab10",
    s=100,
    edgecolor="black",
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Model 1")
plt.ylabel("p-value")
plt.yscale("log")  # Use log scale for p-values
plt.title("p-values for Model Comparisons")
plt.legend(title="Model 2", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("p_values_plot.png", dpi=600)
plt.show()
