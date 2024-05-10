import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import tritonclient.http as httpclient
from tritonclient.utils import *
import time
import threading

stop_processing=False
blip_vqa_visual_encoder_batch_size=3
blip_vqa_text_encoder_batch_size=3
blip_vqa_text_decoder_batch_size=3

def while_blip_vqa_visual_encoder():
    global stop_processing,blip_vqa_visual_encoder_batch_size
    model_name = "blip_vqa_visual_encoder"
    with httpclient.InferenceServerClient("localhost:8000") as client:
        while not stop_processing:
            input0_data=np.array([b"/workspace/demos/images/merlion.png"]*blip_vqa_visual_encoder_batch_size)
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


def while_blip_vqa_text_encoder():
    global stop_processing,blip_vqa_text_encoder_batch_size
    model_name = "blip_vqa_text_encoder"
    with httpclient.InferenceServerClient("localhost:8000") as client:
        while not stop_processing:
            input0_data = np.repeat(np.load("/workspace/pretrained/images_embeds.npy"),blip_vqa_text_encoder_batch_size,axis=0)
            input1_data = np.array(
                [b"where is the woman sitting?"]*blip_vqa_text_encoder_batch_size
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



def while_blip_vqa_text_decoder():
    global stop_processing,blip_vqa_text_decoder_batch_size
    model_name = "blip_vqa_text_decoder"
    with httpclient.InferenceServerClient("localhost:8000") as client:
        while not stop_processing:
            input0_data = np.repeat(np.load("/workspace/pretrained/questions_states.npy"),blip_vqa_text_decoder_batch_size,axis=0)

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


if __name__=="__main__":
    stop_processing=False
    blip_vqa_visual_encoder_thread = threading.Thread(target=while_blip_vqa_visual_encoder,)
    blip_vqa_visual_encoder_thread.start()
    time.sleep(5)
    blip_vqa_visual_encoder_batch_size=2
    time.sleep(5)
    stop_processing=True
    time.sleep(5)
    blip_vqa_visual_encoder_thread.join()

    stop_processing=False
    blip_vqa_text_encoder_thread = threading.Thread(target=while_blip_vqa_text_encoder,)
    blip_vqa_text_encoder_thread.start()
    time.sleep(5)
    blip_vqa_text_encoder_batch_size=2
    time.sleep(5)
    stop_processing=True
    time.sleep(5)
    blip_vqa_text_encoder_thread.join()

    stop_processing=False
    blip_vqa_text_decoder_thread = threading.Thread(target=while_blip_vqa_text_decoder,)
    blip_vqa_text_decoder_thread.start()
    time.sleep(5)
    blip_vqa_text_decoder_batch_size=2
    time.sleep(5)
    stop_processing=True
    time.sleep(5)
    blip_vqa_text_decoder_thread.join()