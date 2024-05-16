import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

model_name = "blip_vqa"
loop_size = 1

with httpclient.InferenceServerClient("localhost:8000") as client:
    images = np.array([
        b"/workspace/demos/images/beach.jpg",
        b"/workspace/demos/images/beach.jpg",
        b"/workspace/demos/images/merlion.png",
        b"/workspace/demos/images/merlion.png",
    ] * loop_size)
    questions = np.array([
        b"where is the woman sitting?",
        b"where is the dog sitting?",
        b"",
        b"which city is this photo taken?",
    ] * loop_size)
    inputs = [
        httpclient.InferInput(
            "INPUT0",
            images.shape,
            np_to_triton_dtype(images.dtype),
        ),
        httpclient.InferInput(
            "INPUT1",
            questions.shape,
            np_to_triton_dtype(questions.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(images)
    inputs[1].set_data_from_numpy(questions)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    answers = response.as_numpy("OUTPUT0")

    print("IMAGE ({}) + QUESTION ({}) = ANSWER ({})".format(
        images, questions, answers))

