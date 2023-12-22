import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

model_name = "blip_vqa"
loop_size = 1
USE_MODAL_LEVEL_BATCH = False

with httpclient.InferenceServerClient("localhost:8000") as client:
    images = np.array([
        b"/workspace/examples/images/beach.jpg",
        b"/workspace/examples/images/beach.jpg",
        b"/workspace/examples/images/merlion.png",
        b"/workspace/examples/images/merlion.png",
    ] * loop_size)
    questions = np.array([
        b"where is the woman sitting?",
        b"where is the dog sitting?",
        b"",
        b"which city is this photo taken?",
    ] * loop_size)
    use_modal_level_batch = np.array([
        USE_MODAL_LEVEL_BATCH,
        USE_MODAL_LEVEL_BATCH,
        USE_MODAL_LEVEL_BATCH,
        USE_MODAL_LEVEL_BATCH,
    ] * loop_size)

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
        httpclient.InferInput(
            "USE_MODAL_LEVEL_BATCH",
            use_modal_level_batch.shape,
            np_to_triton_dtype(use_modal_level_batch.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(images)
    inputs[1].set_data_from_numpy(questions)
    inputs[2].set_data_from_numpy(use_modal_level_batch)

    outputs = [
        httpclient.InferRequestedOutput("ANSWER"),
    ]

    for _ in range(7):
        response = client.infer(model_name,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)
        result = response.get_response()

    answers = response.as_numpy("ANSWER")

    print("IMAGE ({}) + QUESTION ({}) = ANSWER ({})".format(
        images, questions, answers))

    sys.exit(0)
