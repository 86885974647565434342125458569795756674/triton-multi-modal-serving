from PIL import Image
import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import sys
import os

root_path='/dynamic_batch/triton-multi-modal-serving'

sys.path.append(root_path)
from models.blip.blip_vqa_text_decoder import blip_vqa_text_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
questions_states = np.load(root_path+"/pretrained/questions_states.npy")
print(questions_states.shape)

questions_atts_shape=(questions_states.shape[0]*questions_states.shape[1],questions_states.shape[2])
questions_atts = torch.ones(questions_atts_shape, dtype=torch.long).numpy(force=True)
print(questions_atts.shape)

model_url = root_path+"/pretrained/model_base_vqa_capfilt_large.pth"

model = blip_vqa_text_decoder(pretrained=model_url, vit="base")
model.eval()
print(sum(p.numel() for p in model.parameters()))
model = model.to(device)

with torch.no_grad():
    answers = model(questions_states,questions_atts)
    #answers = model(questions_states)
print(answers)
print(answers.shape)
