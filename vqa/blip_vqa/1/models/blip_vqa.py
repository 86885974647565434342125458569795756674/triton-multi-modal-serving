from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import numpy.ma as ma
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

        self.image_size = image_size
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1
        )

        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

    def forward(self, image_urls, questions, enable_modal_level_batch=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image Preprocesor
        start = time.time()
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        image_urls = [image_url.decode() for image_url in image_urls]
        images = []
        if enable_modal_level_batch:
            for image_url in image_urls:
                if image_url != "":
                    images.append(transform(Image.open(image_url).convert("RGB")))
        else:
            for image_url in image_urls:
                if image_url == "":
                    images.append(images[-1].clone())
                else:
                    images.append(transform(Image.open(image_url).convert("RGB")))
        images = torch.stack(images)
        end = time.time()
        print(f"[image batch size: {images.size(0)}]")
        image_preprocessor_time = end - start
        print("image preprocessor time: ", image_preprocessor_time)

        # Visual Encoder
        start = time.time()
        images = images.to(device)
        images_embeds = self.visual_encoder(images)
        torch.cuda.synchronize()
        end = time.time()
        image_encoder_time = end - start
        print("image encoder time: ", image_encoder_time)

        # Text Tokenizer
        question_batch_size = questions.shape[0]
        print(f"[question batch size: {question_batch_size}]")
        start = time.time()
        questions = self.tokenizer(
            [question.decode() for question in questions],
            padding="longest",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        end = time.time()
        text_tokenizer_time = end - start
        print("text tokenizer time: ", text_tokenizer_time)

        # Image Autoscaler
        image_autoscaler_time = 0.0
        if enable_modal_level_batch:
            start = time.time()
            image_embeds_idx = -1
            a_images_embeds = []
            for image_url in image_urls:
                if image_url != "":
                    image_embeds_idx += 1
                a_images_embeds.append(images_embeds[image_embeds_idx])
            images_embeds = torch.stack(a_images_embeds)
            end = time.time()
            image_autoscaler_time = end - start
            print("image autoscaler time: ", image_autoscaler_time)

        # Text Encoder
        start = time.time()
        images_atts = torch.ones(images_embeds.size()[:-1], dtype=torch.long).to(device)
        questions = questions.to(device)
        questions_output = self.text_encoder(
            questions.input_ids,
            attention_mask=questions.attention_mask,
            encoder_hidden_states=images_embeds,
            encoder_attention_mask=images_atts,
            return_dict=True,
        )
        num_beams = 3
        questions_states = questions_output.last_hidden_state.repeat_interleave(
            num_beams, dim=0
        )
        torch.cuda.synchronize()
        end = time.time()
        text_encoder_time = end - start
        print("text encoder time: ", text_encoder_time)

        # Text Decoder
        start = time.time()
        questions_atts = torch.ones(questions_states.size()[:-1], dtype=torch.long).to(
            device
        )
        model_kwargs = {
            "encoder_hidden_states": questions_states,
            "encoder_attention_mask": questions_atts,
        }
        bos_ids = torch.full(
            (question_batch_size, 1),
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
            **model_kwargs,
        )

        answers = [
            self.tokenizer.decode(output, skip_special_tokens=True).encode()
            for output in outputs
        ]
        end = time.time()
        text_decoder_time = end - start
        print("text decoder time: ", text_decoder_time)
        print("")

        with open("/workspace/result.txt", "a") as f:
            print(
                image_preprocessor_time,
                image_encoder_time,
                text_tokenizer_time,
                image_autoscaler_time,
                text_encoder_time,
                text_decoder_time,
                sep="\t",
                file=f,
            )

        return np.array(answers)


def blip_vqa(pretrained="", **kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    return model
