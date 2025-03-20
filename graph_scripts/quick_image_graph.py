import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

# Define the directories
ok_front_dir = r"../casting_512x512/casting_512x512/ok_front"
def_front_dir = r"../casting_512x512/casting_512x512/def_front"

# Get a list of all files in each directory
ok_front_images = [
    f for f in os.listdir(ok_front_dir) if os.path.isfile(os.path.join(ok_front_dir, f))
]
def_front_images = [
    f
    for f in os.listdir(def_front_dir)
    if os.path.isfile(os.path.join(def_front_dir, f))
]

# Pick a random image from each directory
ok_front_image = random.choice(ok_front_images)
def_front_image = random.choice(def_front_images)

# Create the full path to the images
ok_front_path = os.path.join(ok_front_dir, ok_front_image)
def_front_path = os.path.join(def_front_dir, def_front_image)

# Load the images
img_ok = mpimg.imread(ok_front_path)
img_def = mpimg.imread(def_front_path)

# Create the figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# Plot the ok_front image
axes[0].imshow(img_ok)
axes[0].set_title("Non-Defective", fontsize=16)
axes[0].axis("off")  # Turn off axis labels

# Plot the def_front image
axes[1].imshow(img_def)
axes[1].set_title("Defective", fontsize=16)
axes[1].axis("off")  # Turn off axis labels

# Set the title of the figure
fig.suptitle("Sample Casting Images", fontsize=20, fontweight="bold")

# Save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
plt.savefig("sample_images_larger_font.png", dpi=600)

# Show the plot (optional)
plt.show()
