import json

import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

dataset_dir = "/datasets/vqa/"
json_file = dataset_dir + "test.json"
with open(json_file) as f:
    dataset = json.load(f)

model_name = "blip_vqa"
# sample_size = 128
sample_size = 16
batch_size = 16
USE_MODAL_LEVEL_BATCH = True

with httpclient.InferenceServerClient("localhost:8000") as client:
    image_batch = []
    question_batch = []
    use_modal_level_batch = np.array([USE_MODAL_LEVEL_BATCH] * batch_size)
    for data in dataset[0:sample_size]:
        image_batch.append(bytes(dataset_dir + data["image"], "utf-8"))
        question_batch.append(bytes(data["question"], "utf-8"))
        if len(image_batch) == batch_size:
            images = np.array(image_batch)
            questions = np.array(question_batch)
            image_batch = []
            question_batch = []

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

            response = client.infer(model_name,
                                    inputs,
                                    request_id=str(1),
                                    outputs=outputs)
            result = response.get_response()

            answers = response.as_numpy("ANSWER")
            print("IMAGE ({}) + QUESTION ({}) = ANSWER ({})".format(
                images, questions, answers))
    sys.exit(0)
