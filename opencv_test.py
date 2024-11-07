from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from urllib.request import urlopen
from PIL import Image
import timm
import torch

with open('image_net_labels', 'r') as f:
    content = f.read()

image_net_labels = eval(content)

model = timm.create_model('mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k', pretrained = True)
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training = False)

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    output = model(transforms(img).insqueeze(0))

    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k = 5)

    print("Top 5")
    print(top5_probabilities)
    print(top5_class_indices)

    predictions = []
    for i in range(5)
        try:
            predictions.append([image_net_labels
