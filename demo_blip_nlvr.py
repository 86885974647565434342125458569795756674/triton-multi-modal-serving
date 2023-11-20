import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "blip_nlvr"
loop_size = 4

with httpclient.InferenceServerClient("localhost:8000") as client:
    image0_urls = np.array(
        [
            b"/workspace/examples/ex0_0.jpg",
            b"/workspace/examples/acorns_1.jpg",
        ]
        * loop_size
    )
    image1_urls = np.array(
        [
            b"/workspace/examples/ex0_1.jpg",
            b"/workspace/examples/acorns_6.jpg",
        ]
        * loop_size
    )
    texts = np.array(
        [
            b"The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.",
            b"One image shows exactly two brown acorns in back-to-back caps on green foliage.",
        ]
        * loop_size
    )

    inputs = [
        httpclient.InferInput(
            "IMAGE0",
            image0_urls.shape,
            np_to_triton_dtype(image0_urls.dtype),
        ),
        httpclient.InferInput(
            "IMAGE1",
            image1_urls.shape,
            np_to_triton_dtype(image1_urls.dtype),
        ),
        httpclient.InferInput(
            "TEXT",
            texts.shape,
            np_to_triton_dtype(texts.dtype),
        ),
    ]

    inputs[0].set_data_from_numpy(image0_urls)
    inputs[1].set_data_from_numpy(image1_urls)
    inputs[2].set_data_from_numpy(texts)

    outputs = [
        httpclient.InferRequestedOutput("ANSWER"),
    ]

    for _ in range(7):
        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
        result = response.get_response()

    answers = response.as_numpy("ANSWER")

    print(
        "IMAGE0 ({}) + IMAGE1 ({}) + TEXT ({})= ANSWER ({})".format(
            image0_urls, image1_urls, texts, answers
        )
    )

    sys.exit(0)
