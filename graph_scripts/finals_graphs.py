import matplotlib.pyplot as plt
import seaborn as sns

# --- Data ---
model_data = {"MobileNetV2": 6, "MobileNetV3 Small": 5}
tuning_data = {"Fine Tune": 10, "Transfer Learning": 1}
quantization_data = {"PTDQ": 5, "Original": 6}

# --- Graphing Functions ---
colors = ["#7A41BE", "#BE7A41", "#41BE7A"]


def create_bar_graph(data, title, x_label, y_label, filename):
    """Creates a bar graph from the given data."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        data.keys(), data.values(), color=[colors[0], colors[2]], edgecolor="black"
    )
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=0, ha="center")
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            int(yval),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()


def create_pie_chart(data, title, filename):
    """Creates a pie chart from the given data."""
    plt.figure(figsize=(6, 6))
    plt.pie(
        data.values(),
        labels=data.keys(),
        autopct="%1.1f%%",
        startangle=140,
        colors=[colors[0], colors[2]],
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()


# --- Generate Graphs ---
create_bar_graph(
    model_data,
    "Final Models Reaching >= 99.5% Testing Accuracy",
    "Model",
    "Count",
    "model_usage_bar.png",
)
create_pie_chart(
    tuning_data,
    "Tuning Method of Final Models",
    "tuning_distribution_pie.png",
)
create_pie_chart(
    quantization_data,
    "Quantization Type of Final Models",
    "quantization_distribution_pie.png",
)

print("Graphs generated successfully!")
