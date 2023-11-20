import numpy as np
import torch


from models.blip_nlvr import blip_nlvr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image0_urls = np.array(
    [
        b"/workspace/examples/ex0_0.jpg",
        b"/workspace/examples/acorns_1.jpg",
    ]
)
image1_urls = np.array(
    [
        b"/workspace/examples/ex0_1.jpg",
        b"/workspace/examples/acorns_6.jpg",
    ]
)
print(image0_urls)
print(image1_urls)
texts = np.array(
    [
        b"The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.",
        b"One image shows exactly two brown acorns in back-to-back caps on green foliage.",
    ]
)
print(texts)


# model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth"
model_url = "/workspace/model_base_nlvr.pth"

model = blip_nlvr(pretrained=model_url)
model.eval()
model = model.to(device)

with torch.no_grad():
    predictions = model(image0_urls, image1_urls, texts)
print(predictions)
