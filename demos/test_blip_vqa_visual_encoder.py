import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "blip_vqa_visual_encoder"
batch_size = 1

def load_example_image(image_size):
    raw_image = Image.open("/workspace/demos/images/merlion.png").convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(raw_image)
    return image

image_size = 480
image2 = load_example_image(image_size=image_size)
#print(image2.shape,image2.dtype)
images=torch.cat([image2]*batch_size).reshape(batch_size,*image2.shape).numpy()
#print(images.shape, images.dtype)
#torch.Size([3, 480, 480]) torch.float32
#(2, 3, 480, 480) float32


with httpclient.InferenceServerClient("localhost:8000") as client:
    # image = np.load("image1.npy")
    # input0_data = np.repeat([image], [batch_size], axis=0)
    # input1_data = np.repeat([[b"where is the woman sitting?"]], [batch_size], axis=0)

    input0_data=images
    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()

    output0_data = response.as_numpy("OUTPUT0")

    print("INPUT0 ({})  = OUTPUT0 ({})".format(input0_data.shape, output0_data.shape))
