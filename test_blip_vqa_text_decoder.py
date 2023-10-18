import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

batch_size = 1

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.load("questions_states.npy")

    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(
        "blip_vqa_text_decoder", inputs, request_id=str(1), outputs=outputs
    )

    result = response.get_response()

    output0_data = response.as_numpy("OUTPUT0")

    print("INPUT0 {} = OUTPUT0 {}".format(input0_data.shape, output0_data))

    sys.exit(0)
