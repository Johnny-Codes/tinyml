from urllib.request import urlopen
from PIL import Image
import timm
import torch

with open('image_net_labels', 'r') as f:
    dict_string = f.read()

image_net_labels = eval(dict_string)

img_url = "https://hatrabbits.com/wp-content/uploads/2017/01/random.jpg"
#'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
img = Image.open(urlopen(img_url))

model = timm.create_model('mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

#print("==================== OUTPUT =================")
#print(output)
#print("=============================================")
print("=============top5============================")
print(top5_probabilities)
print(top5_class_indices)
print("=============================================")

top5_class = top5_class_indices[0]
list_form_c = top5_class.tolist()

top5_prob = top5_probabilities[0]
list_form_p = top5_prob.tolist()

predictions = []
for i in range(5):
    try:
        predictions.append([image_net_labels[list_form_c[i]], round(list_form_p[i], 2)])
    except KeyError:
        print(f"error for {i} - {list_form_p[i]}") 
print("--------------- PREDICTIONS ------------------")
print(predictions)
