import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "blip_vqa"

with httpclient.InferenceServerClient("localhost:8000") as client:
    image = np.load("image.npy")
    input0_data = np.repeat(image, [4], axis=0)
    input1_data = np.full((4,), b"where is the woman sitting?")

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

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()

    output0_data = response.as_numpy("OUTPUT0")

    print(
        "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
            input0_data.shape, input1_data, output0_data
        )
    )

    sys.exit(0)
