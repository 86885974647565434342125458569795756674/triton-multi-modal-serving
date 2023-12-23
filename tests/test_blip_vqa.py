import os
import sys

sys.path.append(os.path.join(os.getcwd()))

import numpy as np
import torch

from models import blip_vqa

image_urls = np.array([
    b"/workspace/examples/images/beach.jpg",
    b"/workspace/examples/images/beach.jpg",
    b"/workspace/examples/images/merlion.png",
    b"/workspace/examples/images/merlion.png",
])
questions = np.array([
    b"where is the woman sitting?",
    b"where is the dog sitting?",
    b"",
    b"which city is this photo taken?",
])

model_url = "/pretrained/model_base_vqa_capfilt_large.pth"
# model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = blip_vqa(pretrained=model_url)
model.eval()
model = model.to(device)

with torch.no_grad():
    answers = model(image_urls, questions, enable_modal_level_batch=True)
with torch.no_grad():
    answers = model(image_urls, questions, enable_modal_level_batch=False)
print(image_urls)
print(questions)
print(answers)
print(answers)
