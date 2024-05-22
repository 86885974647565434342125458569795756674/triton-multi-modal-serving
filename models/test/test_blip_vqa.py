import os
import sys


import numpy as np
import torch

sys.path.append(os.path.join(os.getcwd()))
from models.blip.blip_vqa import blip_vqa

image_urls = np.array([
    b"/workspace/demos/images/beach.jpg",
    b"/workspace/demos/images/beach.jpg",
    b"/workspace/demos/images/merlion.png",
    b"/workspace/demos/images/merlion.png",
])
questions = np.array([
    b"where is the woman sitting?",
    b"where is the dog sitting?",
    b"",
    b"which city is this photo taken?",
])

model_url = "/workspace/pretrained/model_base_vqa_capfilt_large.pth"
# model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = blip_vqa(pretrained=model_url)
model.eval()
model = model.to(device)

with torch.no_grad():
    answers = model(image_urls, questions)
print(image_urls)
print(questions)
print(answers)