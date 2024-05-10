from models.blip.med import BertConfig, BertModel, BertLMHeadModel
from models.blip.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import time


class BLIP_VQA_VISUAL_ENCODER(nn.Module):
    def __init__(
        self,
        med_config="configs/med_config.json",
        image_size=480,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1
        )

    def forward(self, images):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("batch size:", images.shape[0])

        # Visual Encoder
        start = time.time()

        images = torch.from_numpy(images).to(device)
        images_embeds = self.visual_encoder(images)
        images_embeds = images_embeds.numpy(force=True)#to(cpu)

        end = time.time()
        print("visual_encoder time:", end - start)

        return images_embeds


def blip_vqa_visual_encoder(pretrained="", **kwargs):
    model = BLIP_VQA_VISUAL_ENCODER(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    return model
