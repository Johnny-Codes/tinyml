import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load original images
img2_boxes_color = cv.imread("./boxes_split/test/damaged/0101069901524_top.png")
img4_boxes_color = cv.imread("./boxes_split/test/intact/0407105940330_top.png")
assert img2_boxes_color is not None and img4_boxes_color is not None, "no file"


def preprocess_image(img_color):
    # Convert to Grayscale
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

    # Apply Adaptive Thresholding
    img_thresh = cv.adaptiveThreshold(
        img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )

    # Apply Canny Edge Detection
    edges = cv.Canny(img_thresh, 100, 200)

    # Find Contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangular contour
    largest_contour = max(contours, key=cv.contourArea)
    rect = cv.minAreaRect(largest_contour)
    box = cv.boxPoints(rect)
    box = np.intp(box)  # Use np.intp instead of np.int0

    # Draw the bounding box
    img_box = img_color.copy()
    cv.drawContours(img_box, [box], 0, (0, 0, 255), 2)

    # Apply Perspective Transformation or Crop the Bounding Box
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(img_color, M, (width, height))

    # Apply GrabCut for Background Removal
    mask = np.zeros(warped.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, width - 2, height - 2)
    cv.grabCut(warped, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Ensure the mask and warped image have the same dimensions
    mask2 = cv.resize(mask2, (warped.shape[1], warped.shape[0]))

    img_bg_removed = warped * mask2[:, :, np.newaxis]

    return img_box, img_thresh, edges, img_bg_removed


# Process the box images
img2_box, img2_thresh, img2_edges, img2_bg_removed = preprocess_image(img2_boxes_color)
img4_box, img4_thresh, img4_edges, img4_bg_removed = preprocess_image(img4_boxes_color)

# Create subplots
plt.figure(figsize=(22, 12))

# Display the results for the first box image
plt.subplot(4, 4, 1), plt.imshow(cv.cvtColor(img2_boxes_color, cv.COLOR_BGR2RGB))
plt.title("Original Image 1"), plt.xticks([]), plt.yticks([])

plt.subplot(4, 4, 2), plt.imshow(img2_thresh, cmap="gray")
plt.title("Threshold Image 1"), plt.xticks([]), plt.yticks([])

plt.subplot(4, 4, 3), plt.imshow(img2_edges, cmap="gray")
plt.title("Edge Image 1"), plt.xticks([]), plt.yticks([])

plt.subplot(4, 4, 4), plt.imshow(cv.cvtColor(img2_bg_removed, cv.COLOR_BGR2RGB))
plt.title("BG Removed 1"), plt.xticks([]), plt.yticks([])

# Display the results for the second box image
plt.subplot(4, 4, 5), plt.imshow(cv.cvtColor(img4_boxes_color, cv.COLOR_BGR2RGB))
plt.title("Original Image 2"), plt.xticks([]), plt.yticks([])

plt.subplot(4, 4, 6), plt.imshow(img4_thresh, cmap="gray")
plt.title("Threshold Image 2"), plt.xticks([]), plt.yticks([])

plt.subplot(4, 4, 7), plt.imshow(img4_edges, cmap="gray")
plt.title("Edge Image 2"), plt.xticks([]), plt.yticks([])

plt.subplot(4, 4, 8), plt.imshow(cv.cvtColor(img4_bg_removed, cv.COLOR_BGR2RGB))
plt.title("BG Removed 2"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
