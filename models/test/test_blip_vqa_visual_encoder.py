from PIL import Image
import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import sys

sys.path.append(os.path.join(os.getcwd()))
from models.blip.blip_vqa_visual_encoder import blip_vqa_visual_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_example_image(image_size):
    raw_image = Image.open("/workspace/demos/images/merlion.png").convert("RGB")
    print(type(raw_image))
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(raw_image)
    return image

#image_size = 480
#image2 = load_example_image(image_size=image_size)
#print(image2.shape,image2.dtype)
#images=torch.cat([image2,image2]).reshape(2,*image2.shape).numpy()
#print(images.shape, images.dtype)
#torch.Size([3, 480, 480]) torch.float32
#(2, 3, 480, 480) float32
images=np.array([b"/workspace/demos/images/merlion.png"]*1)
model_url = "/workspace/pretrained/model_base_vqa_capfilt_large.pth"

model = blip_vqa_visual_encoder(pretrained=model_url, vit="base")

model.eval()
model = model.to(device)

images_embeds = model(images)

#print(images_embeds.shape,images_embeds.dtype)
#(2, 901, 768) float32

with open("/workspace/pretrained/images_embeds.npy", "wb") as f:
    np.save(f, images_embeds)
