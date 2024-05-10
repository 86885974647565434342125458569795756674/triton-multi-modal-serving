from PIL import Image
import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from models.blip.blip_vqa_text_decoder import blip_vqa_text_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
questions_states = np.load("/workspace/pretrained/questions_states.npy")
print(questions_states.shape)

model_url = "/workspace/pretrained/model_base_vqa_capfilt_large.pth"

model = blip_vqa_text_decoder(pretrained=model_url, vit="base")
model.eval()
model = model.to(device)

with torch.no_grad():
    answers = model(questions_states)
print(answers)
print(answers.shape)
