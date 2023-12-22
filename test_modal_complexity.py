import sys
import os

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

MODEL_NAME = "blip_vqa"
USE_MODAL_LEVEL_BATCH = False


def inference(batch_size):
    images = np.array([
        b"/workspace/examples/beach.jpg",
    ] * batch_size)
    questions = np.array([
        b"where is the woman sitting?",
    ] * batch_size)
    use_modal_level_batch = np.array([
        USE_MODAL_LEVEL_BATCH,
    ] * batch_size)

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
    client.infer(MODEL_NAME, inputs, request_id=str(1), outputs=outputs)


os.remove("output.txt")
with httpclient.InferenceServerClient("localhost:8000") as client:
    max_batch_size = 64
    batch_size_list = [1] + list(range(2, max_batch_size + 1, 2))
    for batch_size in batch_size_list:
        inference(batch_size)
        for _ in range(max_batch_size // batch_size):
            inference(batch_size)
    sys.exit(0)
