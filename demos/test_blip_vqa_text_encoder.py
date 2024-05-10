import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

batch_size = 1

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.load("/workspace/pretrained/images_embeds.npy")
    input1_data = np.array(
        [b"where is the woman sitting?",b"which city is this photo taken?"]
    )

    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        httpclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(
        "blip_vqa_text_encoder", inputs, request_id=str(1), outputs=outputs
    )

    result = response.get_response()

    output0_data = response.as_numpy("OUTPUT0")

    print(
        "INPUT0 {} + INPUT1 {} = OUTPUT0 {}".format(
            input0_data.shape, input1_data, output0_data.shape
        )
    )

