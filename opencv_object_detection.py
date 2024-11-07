import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

# Load a pre-trained Faster R-CNN model with the most up-to-date weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Define the image preprocessing pipeline
preprocess = T.Compose(
    [
        T.ToPILImage(),
        T.ToTensor(),
    ]
)

# COCO dataset labels
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)
# print("Before while loop")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print("Frame captured")

    # If frame is read correctly, ret is True
    if not ret:
        # print("Failed to grab frame")
        break

    # Preprocess the frame
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(
        0
    )  # Create a mini-batch as expected by the model

    # Move the input and model to CPU
    input_batch = input_batch.to("cpu")
    model.to("cpu")

    with torch.no_grad():
        predictions = model(input_batch)

    # print(f"Predictions: {predictions}")  # Debug print to check if predictions are made

    # Check if predictions are empty
    if len(predictions[0]["boxes"]) == 0:
        print("No predictions made")

    # Draw bounding boxes and labels on the frame
    max_objects = 3
    for idx, box in enumerate(predictions[0]["boxes"]):
        if idx >= max_objects:
            break
        if predictions[0]["scores"][idx] > 0.5:  # Higher confidence threshold
            boxes = box.cpu().numpy().astype(int)
            label_idx = predictions[0]["labels"][idx].item()
            label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
            score = predictions[0]["scores"][idx].item()
            # print(
            #     f"Box coordinates: {boxes}, Label: {label}, Score: {score:.2f}"
            # )  # Debug print

            # Ensure the bounding box coordinates are within the frame dimensions
            x1, y1, x2, y2 = boxes
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"Label: {label}, Score: {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

    # Display the resulting frame
    cv2.imshow("Webcam", frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
