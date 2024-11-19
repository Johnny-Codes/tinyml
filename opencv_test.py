import cv2
from PIL import Image
import timm
import torch
import torchvision.transforms as transforms

# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

with open("image_net_labels", "r") as f:
    file_contents = f.read()

image_net_labels = eval(file_contents)

model = timm.create_model(
    "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k", pretrained=True
)
model = model.eval()

# Define the image preprocessing pipeline
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(
        0
    )  # Create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Display the predictions on the frame
    for i in range(top5_prob.size(0)):
        label = image_net_labels[top5_catid[i].item()]
        prob = top5_prob[i].item()
        cv2.putText(
            frame,
            f"{label}: {prob:.2f}",
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    # Display the resulting frame
    cv2.imshow("Dickbutt", frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
