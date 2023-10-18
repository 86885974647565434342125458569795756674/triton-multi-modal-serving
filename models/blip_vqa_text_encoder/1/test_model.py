import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


images_embeds = np.load("/workspace/images_embeds.npy")

print(images_embeds.shape)
questions = np.array(
    [[b"where is the woman sitting?"], [b"which city is this photo taken?"]]
)
# questions = np.full((batch_size,), b"where is the woman sitting?")
print(questions)


model_url = "/workspace/model_base_vqa_capfilt_large.pth"
# model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"

from models.blip_vqa_text_encoder import blip_vqa_text_encoder

model = blip_vqa_text_encoder(pretrained=model_url, vit="base")
# model = torch.load("blip_vqa.pt")
model.eval()
model = model.to(device)


questions_states = model(images_embeds, questions)

with open("/workspace/questions_states.npy", "wb") as f:
    np.save(f, questions_states)

print(questions_states.shape)
print(questions_states)

import shutil

shutil.copyfile(
    r"/workspace/models/blip_vqa_text_encoder/1/models/blip_vqa_text_encoder.py",
    r"/workspace/examples/blip_vqa/1/models/blip_vqa_text_encoder.py",
)
shutil.copyfile(
    r"/workspace/models/blip_vqa_text_encoder/1/models/blip_vqa_text_encoder.py",
    r"/workspace/models/blip_vqa_text_decoder/1/models/blip_vqa_text_encoder.py",
)
shutil.copyfile(
    r"/workspace/models/blip_vqa_text_encoder/1/models/blip_vqa_text_encoder.py",
    r"/workspace/models/blip_vqa_visual_encoder/1/models/blip_vqa_text_encoder.py",
)
