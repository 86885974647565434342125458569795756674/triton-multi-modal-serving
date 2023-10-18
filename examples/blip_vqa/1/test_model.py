from PIL import Image
import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_demo_image(image_size):
    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    w, h = raw_image.size

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


def load_example_image(image_size):
    raw_image = Image.open("/workspace/merlion.png").convert("RGB")

    w, h = raw_image.size

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


from models.blip_vqa import blip_vqa

# image1 = load_demo_image(image_size=image_size)
# image2 = load_example_image(image_size=image_size)
# with open("/workspace/image1.npy", "wb") as f:
#     np.save(f, image1)
# with open("/workspace/image2.npy", "wb") as f:
#     np.save(f, image2)
image1 = np.load("/workspace/image1.npy")
image2 = np.load("/workspace/image2.npy")


images = np.array([image1, image2])
print(images.shape, images.dtype)
questions = np.array(
    [[b"where is the woman sitting?"], [b"which city is this photo taken?"]]
)
# questions = np.full((batch_size,), b"where is the woman sitting?")
print(questions)


model_url = "/workspace/model_base_vqa_capfilt_large.pth"
# model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"

model = blip_vqa(pretrained=model_url, vit="base")
# model = torch.load("blip_vqa.pt")
model.eval()
model = model.to(device)

with torch.no_grad():
    answers = model(images, questions)
print(answers)

import shutil

shutil.copyfile(
    r"/workspace/examples/blip_vqa/1/models/blip_vqa.py",
    r"/workspace/models/blip_vqa_text_decoder/1/models/blip_vqa.py",
)
shutil.copyfile(
    r"/workspace/examples/blip_vqa/1/models/blip_vqa.py",
    r"/workspace/models/blip_vqa_text_encoder/1/models/blip_vqa.py",
)
shutil.copyfile(
    r"/workspace/examples/blip_vqa/1/models/blip_vqa.py",
    r"/workspace/models/blip_vqa_visual_encoder/1/models/blip_vqa.py",
)
