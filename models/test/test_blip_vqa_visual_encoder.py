from PIL import Image
import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import sys

root_path='/dynamic_batch/triton-multi-modal-serving'

sys.path.append(root_path)
from models.blip.blip_vqa_visual_encoder import blip_vqa_visual_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#image_size = 480
#image2 = load_example_image(image_size=image_size)
#print(image2.shape,image2.dtype)
#images=torch.cat([image2,image2]).reshape(2,*image2.shape).numpy()
#print(images.shape, images.dtype)
#torch.Size([3, 480, 480]) torch.float32
#(2, 3, 480, 480) float32
images=np.array([[root_path.encode('utf-8')+b"/demos/images/merlion.png"]])
model_url = root_path+"/pretrained/model_base_vqa_capfilt_large.pth"

model = blip_vqa_visual_encoder(pretrained=model_url, vit="base")

model.eval()
print(sum(p.numel() for p in model.parameters()))
model = model.to(device)

with torch.no_grad():
    images_embeds = model(images)

print(images_embeds.shape,images_embeds.dtype)
#(2, 901, 768) float32

with open(root_path+"/pretrained/images_embeds.npy", "wb") as f:
    np.save(f, images_embeds)
