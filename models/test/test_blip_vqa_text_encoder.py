import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import sys
import os

root_path='/dynamic_batch/triton-multi-modal-serving'

sys.path.append(root_path)
from models.blip.blip_vqa_text_encoder import blip_vqa_text_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


images_embeds = np.load(root_path+"/pretrained/images_embeds.npy")

questions = np.array(
    [[b"where is the woman sitting?"]]
)
# questions = np.full((batch_size,), b"where is the woman sitting?")
#print(questions)
#print(questions.size)

model_url = root_path+"/pretrained/model_base_vqa_capfilt_large.pth"

model = blip_vqa_text_encoder(pretrained=model_url, vit="base")


model.eval()
print(sum(p.numel() for p in model.parameters()))
model = model.to(device)

with torch.no_grad():
     questions_states = model(images_embeds, questions)
print(questions_states.shape,questions_states.dtype)
#(2, 1, 9, 768) float32
with open(root_path+"/pretrained/questions_states.npy", "wb") as f:
     np.save(f, questions_states)


