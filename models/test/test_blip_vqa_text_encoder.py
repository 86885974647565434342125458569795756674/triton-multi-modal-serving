import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from models.blip.blip_vqa_text_encoder import blip_vqa_text_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


images_embeds = np.load("/workspace/pretrained/images_embeds.npy")

questions = np.array(
    [b"where is the woman sitting?", b"which city is this photo taken?"]
)
questions = np.array(
    [b"where is the woman sitting?"]
)
# questions = np.full((batch_size,), b"where is the woman sitting?")
#print(questions)
#print(questions.size)

model_url = "/workspace/pretrained/model_base_vqa_capfilt_large.pth"

model = blip_vqa_text_encoder(pretrained=model_url, vit="base")


model.eval()
model = model.to(device)


questions_states = model(images_embeds, questions)
#print(questions_states.shape,questions_states.dtype)
#(2, 1, 9, 768) float32
with open("/workspace/pretrained/questions_states.npy", "wb") as f:
     np.save(f, questions_states)


