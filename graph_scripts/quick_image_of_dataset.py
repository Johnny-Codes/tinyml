import os
import glob
import matplotlib.pyplot as plt


def count_images_in_folder(folder_path):
    return len(glob.glob(os.path.join(folder_path, "*.jpeg")))


def get_image_counts(base_dir):
    categories = ["train", "test", "val"]
    labels = ["def_front", "ok_front"]
    counts = {label: {category: 0 for category in categories} for label in labels}

    for category in categories:
        for label in labels:
            folder_path = os.path.join(base_dir, f"g_{category}", label)
            counts[label][category] = count_images_in_folder(folder_path)

    return counts


def plot_image_counts(counts, filename):
    categories = ["train", "test", "val"]
    labels = ["def_front", "ok_front"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Colors for train, test, val

    fig, ax = plt.subplots()

    for i, label in enumerate(labels):
        bottom = 0
        for j, category in enumerate(categories):
            count = counts[label][category]
            ax.bar(
                label,
                count,
                bottom=bottom,
                color=colors[j],
                label=category if i == 0 else "",
            )
            # Add text annotation for the count
            ax.text(
                label,
                bottom + count / 2,
                str(count),
                ha="center",
                va="center",
                color="white",
                fontsize=16,
                fontweight="bold",
            )
            bottom += count
    title = input("What is the title of the plot? ")
    ax.set_xlabel("Labels", fontsize=20)
    ax.set_ylabel("Number of Images", fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=12)

    plt.savefig(filename, dpi=600)
    plt.show()


if __name__ == "__main__":
    dataset = int(
        input("Which dataset are you interested in? \n1. Casting \n2. Box \n>>> ")
    )
    if dataset == 1:
        base_dir = "./casting_data/casting_data"
    if dataset == 2:
        base_dir = "./"
    counts = get_image_counts(base_dir)
    plot_image_counts(counts, filename="casting_data_set_split_larger_font.png")
