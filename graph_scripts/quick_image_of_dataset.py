import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def count_images_in_folder(folder_path):
    return len(glob.glob(os.path.join(folder_path, "*.jpeg")))


def get_image_counts(base_dir):
    categories = ["train", "test", "val"]
    labels = ["Defective", "Non-Defective"]  # Changed labels here
    counts = {label: {category: 0 for category in categories} for label in labels}

    for category in categories:
        for label in labels:
            # Changed label names in folder path
            folder_path = os.path.join(
                base_dir,
                f"g_{category}",
                "def_front" if label == "Defective" else "ok_front",
            )
            counts[label][category] = count_images_in_folder(folder_path)

    return counts


def plot_image_counts(counts, filename):
    categories = ["train", "test", "val"]
    display_categories = ["Training", "Testing", "Validation"]
    labels = ["Defective", "Non-Defective"]  # Changed labels here
    colors = ["#7A41BE", "#BE7A41", "#41BE7A"]  # Colors for train, test, val

    # Convert counts to a pandas DataFrame for easier plotting with seaborn
    data = []
    for label in labels:
        for i, category in enumerate(categories):
            data.append(
                {
                    "Label": label,
                    "Category": display_categories[i],
                    "Count": counts[label][category],
                }
            )

    df = pd.DataFrame(data)

    # Use seaborn to create a stacked bar plot
    sns.set_theme(style="white")  # Optional: Set a theme
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

    # Create stacked bars
    bottom = None
    for i, category in enumerate(display_categories):
        category_data = df[df["Category"] == category]
        sns.barplot(
            x="Label",
            y="Count",
            data=category_data,
            color=colors[i],
            ax=ax,
            label=category,
            bottom=bottom,
        )
        if bottom is None:
            bottom = category_data["Count"].values
        else:
            bottom = bottom + category_data["Count"].values

    # Add annotations
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(
            x + width / 2,
            y + height / 2,
            f"{int(height)}",
            horizontalalignment="center",
            verticalalignment="center",
            size="large",
            color="white",
            weight="bold",
        )

    # title = input("What is the title of the plot? ")
    title = "Casting Dataset Split between Categories and Labels"
    ax.set_xlabel("Labels", fontsize=16)
    ax.set_ylabel("Number of Images", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.legend(title="Category", fontsize=12)

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(filename, dpi=600)
    plt.show()


if __name__ == "__main__":
    dataset = int(
        input("Which dataset are you interested in? \n1. Casting \n2. Box \n>>> ")
    )
    if dataset == 1:
        base_dir = "../casting_data/casting_data"
    if dataset == 2:
        base_dir = "../"
    counts = get_image_counts(base_dir)
    plot_image_counts(counts, filename="casting_data_set_split_larger_font2.png")
