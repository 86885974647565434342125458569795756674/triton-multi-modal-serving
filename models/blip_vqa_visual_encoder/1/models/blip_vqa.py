from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import time


class BLIP_VQA(nn.Module):
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
        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

    def forward(self, images, questions):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = questions.size
        print("batch size:", batch_size)

        # Visual Encoder
        start = time.time()

        images = torch.from_numpy(images).to(device)
        images_embeds = self.visual_encoder(images)

        end = time.time()
        print("visual_encoder time:", end - start)

        # Text Encoder

        start = time.time()
        image_atts = torch.ones(images_embeds.size()[:-1], dtype=torch.long).to(device)
        questions = self.tokenizer(
            [question[0].decode("utf-8") for question in questions],
            padding="longest",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        questions_output = self.text_encoder(
            questions.input_ids,
            attention_mask=questions.attention_mask,
            encoder_hidden_states=images_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        num_beams = 3
        questions_states = questions_output.last_hidden_state.repeat_interleave(
            num_beams, dim=0
        )

        end = time.time()
        print("text_encoder time:", end - start)

        # Text Decoder
        start = time.time()

        num_beams = 3
        questions_atts = torch.ones(questions_states.size()[:-1], dtype=torch.long).to(
            questions_states.device
        )
        model_kwargs = {
            "encoder_hidden_states": questions_states,
            "encoder_attention_mask": questions_atts,
        }
        bos_ids = torch.full(
            (batch_size, 1),
            fill_value=self.tokenizer.bos_token_id,
            device=device,
        )
        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            max_length=10,
            min_length=1,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )

        answers = [
            [self.tokenizer.decode(output, skip_special_tokens=True).encode()]
            for output in outputs
        ]
        end = time.time()
        print("text_decoder time:", end - start)

        return np.array(answers)


def blip_vqa(pretrained="", **kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    return model
