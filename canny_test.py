import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load original images
img1_color = cv.imread("./casting_data/casting_data/test/def_front/cast_def_0_126.jpeg")
img3_color = cv.imread("./casting_data/casting_data/test/ok_front/cast_ok_0_9.jpeg")
assert img1_color is not None and img3_color is not None, "no file"

# Load grayscale images for Canny edge detection
img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img3 = cv.cvtColor(img3_color, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
img1_blur = cv.GaussianBlur(img1, (5, 5), 0)
img3_blur = cv.GaussianBlur(img3, (5, 5), 0)

# Apply Canny edge detection
edges = cv.Canny(img1_blur, 100, 200)
edges2 = cv.Canny(img3_blur, 100, 200)

# Apply morphological operations to close gaps in edges
kernel = np.ones((5, 5), np.uint8)
edges = cv.dilate(edges, kernel, iterations=1)
edges = cv.erode(edges, kernel, iterations=1)
edges2 = cv.dilate(edges2, kernel, iterations=1)
edges2 = cv.erode(edges2, kernel, iterations=1)

# Blob Detection
params = cv.SimpleBlobDetector_Params()

# Set parameters for blob detection
params.filterByArea = True
params.minArea = 150
params.filterByCircularity = True
params.minCircularity = 0.1
params.filterByConvexity = True
params.minConvexity = 0.87
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(edges)
keypoints2 = detector.detect(edges2)

# Draw detected blobs as red circles
img_with_keypoints = cv.drawKeypoints(
    edges,
    keypoints,
    np.array([]),
    (0, 0, 255),
    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)
img_with_keypoints2 = cv.drawKeypoints(
    edges2,
    keypoints2,
    np.array([]),
    (0, 0, 255),
    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

# Adaptive thresholding
img1_adaptive = cv.adaptiveThreshold(
    edges, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
)
img3_adaptive = cv.adaptiveThreshold(
    edges2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
)

# Ensure the mask is of the correct type and size
img1_adaptive = img1_adaptive.astype(np.uint8)
img3_adaptive = img3_adaptive.astype(np.uint8)

# Apply mask to original images to remove background
img1_bg_removed = cv.bitwise_and(img1_color, img1_color, mask=img1_adaptive)
img3_bg_removed = cv.bitwise_and(img3_color, img3_color, mask=img3_adaptive)

# Boxes data
# Load original images
img2_boxes_color = cv.imread("./boxes_split/test/damaged/0101069901524_top.png")
img4_boxes_color = cv.imread("./boxes_split/test/intact/0407105940330_top.png")
assert img2_boxes_color is not None and img4_boxes_color is not None, "no file"

# Load grayscale images for Canny edge detection
img2_boxes = cv.cvtColor(img2_boxes_color, cv.COLOR_BGR2GRAY)
img4_boxes = cv.cvtColor(img4_boxes_color, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
img2_boxes_blur = cv.GaussianBlur(img2_boxes, (5, 5), 0)
img4_boxes_blur = cv.GaussianBlur(img4_boxes, (5, 5), 0)

# Apply Canny edge detection
edges_boxes = cv.Canny(img2_boxes_blur, 100, 200)
edges2_boxes = cv.Canny(img4_boxes_blur, 100, 200)

# Apply morphological operations to close gaps in edges
edges_boxes = cv.dilate(edges_boxes, kernel, iterations=1)
edges_boxes = cv.erode(edges_boxes, kernel, iterations=1)
edges2_boxes = cv.dilate(edges2_boxes, kernel, iterations=1)
edges2_boxes = cv.erode(edges2_boxes, kernel, iterations=1)

# Detect blobs
keypoints_boxes = detector.detect(edges_boxes)
keypoints2_boxes = detector.detect(edges2_boxes)

# Draw detected blobs as red circles
img_with_keypoints_boxes = cv.drawKeypoints(
    edges_boxes,
    keypoints_boxes,
    np.array([]),
    (0, 0, 255),
    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)
img_with_keypoints2_boxes = cv.drawKeypoints(
    edges2_boxes,
    keypoints2_boxes,
    np.array([]),
    (0, 0, 255),
    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)


# Find the center of the largest blob for cropping
def find_blob_center(keypoints):
    if keypoints:
        largest_blob = max(keypoints, key=lambda kp: kp.size)
        return int(largest_blob.pt[0]), int(largest_blob.pt[1])
    return None, None


center_x_boxes, center_y_boxes = find_blob_center(keypoints_boxes)
center_x2_boxes, center_y2_boxes = find_blob_center(keypoints2_boxes)


# Crop the images based on the blob center
def crop_image(img, center_x, center_y):
    if center_x is not None and center_y is not None:
        x_start = max(center_x - 150, 0)
        x_end = min(center_x + 300, img.shape[1])
        y_start = max(center_y - 100, 0)
        y_end = min(center_y + 200, img.shape[0])
        return img[y_start:y_end, x_start:x_end]
    return img


cropped_edges_boxes = crop_image(edges_boxes, center_x_boxes, center_y_boxes)
cropped_edges2_boxes = crop_image(edges2_boxes, center_x2_boxes, center_y2_boxes)
cropped_img2_boxes_color = crop_image(img2_boxes_color, center_x_boxes, center_y_boxes)
cropped_img4_boxes_color = crop_image(
    img4_boxes_color, center_x2_boxes, center_y2_boxes
)

# Adaptive thresholding
img2_boxes_adaptive = cv.adaptiveThreshold(
    cropped_edges_boxes, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
)
img4_boxes_adaptive = cv.adaptiveThreshold(
    cropped_edges2_boxes, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
)

# Ensure the mask is of the correct type and size
img2_boxes_adaptive = img2_boxes_adaptive.astype(np.uint8)
img4_boxes_adaptive = img4_boxes_adaptive.astype(np.uint8)

# Apply mask to cropped original images to remove background
img2_boxes_bg_removed = cv.bitwise_and(
    cropped_img2_boxes_color, cropped_img2_boxes_color, mask=img2_boxes_adaptive
)
img4_boxes_bg_removed = cv.bitwise_and(
    cropped_img4_boxes_color, cropped_img4_boxes_color, mask=img4_boxes_adaptive
)

# Create subplots
plt.figure(figsize=(22, 12))

# Casting data
plt.subplot(5, 5, 1), plt.imshow(cv.cvtColor(img1_color, cv.COLOR_BGR2RGB))
plt.title("Original Image 1"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 2), plt.imshow(edges, cmap="gray")
plt.title("Edge Image 1"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 3), plt.imshow(img_with_keypoints, cmap="gray")
plt.title("Blobs Image 1"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 4), plt.imshow(img1_adaptive, cmap="gray")
plt.title("Adaptive Threshold Image 1"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 5), plt.imshow(cv.cvtColor(img1_bg_removed, cv.COLOR_BGR2RGB))
plt.title("BG Removed 1"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 6), plt.imshow(cv.cvtColor(img3_color, cv.COLOR_BGR2RGB))
plt.title("Original Image 2"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 7), plt.imshow(edges2, cmap="gray")
plt.title("Edge Image 2"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 8), plt.imshow(img_with_keypoints2, cmap="gray")
plt.title("Blobs Image 2"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 9), plt.imshow(img3_adaptive, cmap="gray")
plt.title("Adaptive Threshold Image 2"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 10), plt.imshow(cv.cvtColor(img3_bg_removed, cv.COLOR_BGR2RGB))
plt.title("BG Removed 2"), plt.xticks([]), plt.yticks([])

# Boxes data
plt.subplot(5, 5, 11), plt.imshow(cv.cvtColor(img2_boxes_color, cv.COLOR_BGR2RGB))
plt.title("Original Image 3"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 12), plt.imshow(edges_boxes, cmap="gray")
plt.title("Edge Image 3"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 13), plt.imshow(img_with_keypoints_boxes, cmap="gray")
plt.title("Blobs Image 3"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 14), plt.imshow(img2_boxes_adaptive, cmap="gray")
plt.title("Adaptive Threshold Image 3"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 15), plt.imshow(cv.cvtColor(img2_boxes_bg_removed, cv.COLOR_BGR2RGB))
plt.title("BG Removed 3"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 16), plt.imshow(cv.cvtColor(img4_boxes_color, cv.COLOR_BGR2RGB))
plt.title("Original Image 4"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 17), plt.imshow(edges2_boxes, cmap="gray")
plt.title("Edge Image 4"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 18), plt.imshow(img_with_keypoints2_boxes, cmap="gray")
plt.title("Blobs Image 4"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 19), plt.imshow(img4_boxes_adaptive, cmap="gray")
plt.title("Adaptive Threshold Image 4"), plt.xticks([]), plt.yticks([])

plt.subplot(5, 5, 20), plt.imshow(cv.cvtColor(img4_boxes_bg_removed, cv.COLOR_BGR2RGB))
plt.title("BG Removed 4"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
