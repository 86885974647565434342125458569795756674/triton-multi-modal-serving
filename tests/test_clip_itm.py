import os
import sys
from torchvision.datasets import CIFAR100

sys.path.append(os.path.join(os.getcwd()))

from models import clip_itm

visual_model = "ViT-B/16"
image_id = 3637
cifar100 = CIFAR100(root="/datasets", download=False, train=False)
_, class_id = cifar100[image_id]
correct_class = cifar100.classes[class_id]
# caption = f"This is a photo of a {correct_class}"
caption = "This is a photo of a dog"

model = clip_itm(visual_model)
results = model.forward(image_id, caption)
print(results)
