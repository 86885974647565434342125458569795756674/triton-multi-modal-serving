from PIL import Image
import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.blip_vqa_text_decoder import blip_vqa_text_decoder

questions_states = np.load("/workspace/questions_states.npy")
print(questions_states.shape)

model_url = "/workspace/model_base_vqa_capfilt_large.pth"
# model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"

model = blip_vqa_text_decoder(pretrained=model_url, vit="base")
model.eval()
model = model.to(device)

with torch.no_grad():
    answers = model(questions_states)
print(answers)

import shutil

shutil.copyfile(
    r"/workspace/models/blip_vqa_text_decoder/1/models/blip_vqa_text_decoder.py",
    r"/workspace/examples/blip_vqa/1/models/blip_vqa_text_decoder.py",
)
shutil.copyfile(
    r"/workspace/models/blip_vqa_text_decoder/1/models/blip_vqa_text_decoder.py",
    r"/workspace/models/blip_vqa_text_encoder/1/models/blip_vqa_text_decoder.py",
)
shutil.copyfile(
    r"/workspace/models/blip_vqa_text_decoder/1/models/blip_vqa_text_decoder.py",
    r"/workspace/models/blip_vqa_visual_encoder/1/models/blip_vqa_text_decoder.py",
)
