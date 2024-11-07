import timm
import torch
from urllib.request import urlopen
from PIL import Image

with open("image_net_labels", "r") as f:
    content = f.read()

image_net_labels = eval(content)

# Load the model
print("I HOPE THIS SHOWS UP")
model = timm.create_model('mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k', pretrained=True)
model.eval()

# Script the model for JIT
print("JITTING")
scripted_model = torch.jit.script(model)
print("print scripted_model done")
# Function to load and preprocess the image
def load_image(img_url):
    print("in load_image")
    img = Image.open(urlopen(img_url))
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return transforms(img).unsqueeze(0)  # Create a batch

# Example image URL
#img_url = "https://hatrabbits.com/wp-content/uploads/2017/01/random.jpg"
#img_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRDdok22lhV69bHnHl4GlHTqq4ZupB8dtN5GQ&s"
#img_url = "https://static.scientificamerican.com/dam/m/4aaa836e513fa8a5/original/krijn_neanderthal_face_reconstruction.jpg?m=1728652157.415&w=600"
#img_url = "https://cdn.openart.ai/stable_diffusion/5ff5ebd766dd0f6022531ac37422944a61adc6a1_2000x2000.webp"
img_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLwsfo9NC1pzBgR2nmp8LIBDIPbZSqODx4Sg&s"
print('image url', img_url)
input_tensor = load_image(img_url)

# Run inference
output = scripted_model(input_tensor)

# Process output
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
print("Top 5 Probabilities:", top5_probabilities)
print("Top 5 Class Indices:", top5_class_indices)

top5_class = top5_class_indices[0]
list_form_c = top5_class.tolist()
top5_prob = top5_probabilities[0]
list_form_p = top5_prob.tolist()

predictions = []
for i in range(5):
    predictions.append([image_net_labels[list_form_c[i]], round(list_form_p[i], 2)])

print(f"PREDICTIONS: {predictions}")
