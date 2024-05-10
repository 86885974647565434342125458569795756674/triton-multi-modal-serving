import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "blip_vqa_visual_encoder"
batch_size = 10

with httpclient.InferenceServerClient("localhost:8000") as client:

    input0_data=np.array([b"/workspace/demos/images/merlion.png"]*batch_size)
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
