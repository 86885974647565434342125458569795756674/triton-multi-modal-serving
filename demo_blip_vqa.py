import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

model_name = "blip_vqa"
loop_size = 64

with httpclient.InferenceServerClient("localhost:8000") as client:
    images = np.array(
        [
            b"/workspace/examples/beach.jpg",
            b"",
            # b"/workspace/examples/merlion.png",
            # b"",
        ]
        * loop_size
    )
    questions = np.array(
        [
            b"where is the woman sitting?",
            b"where is the dog sitting?",
            # b"",
            # b"which city is this photo taken?",
        ]
        * loop_size
    )

    inputs = [
        httpclient.InferInput(
            "IMAGE",
            images.shape,
            np_to_triton_dtype(images.dtype),
        ),
        httpclient.InferInput(
            "QUESTION",
            questions.shape,
            np_to_triton_dtype(questions.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(images)
    inputs[1].set_data_from_numpy(questions)

    outputs = [
        httpclient.InferRequestedOutput("ANSWER"),
    ]

    for _ in range(7):
        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
        result = response.get_response()

    answers = response.as_numpy("ANSWER")

    print("IMAGE ({}) + QUESTION ({}) = ANSWER ({})".format(images, questions, answers))

    sys.exit(0)