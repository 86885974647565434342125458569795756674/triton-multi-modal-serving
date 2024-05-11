import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import tritonclient.http as httpclient
from tritonclient.utils import *
import time
import threading
import queue

def while_client(model_name,input_queue,output_queue):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        while True:
            input_data=input_queue.get()
            if input_data is None:
                break
            inputs=[]
            if isinstance(input_data,tuple):
                for i in range(len(input_data)):
                    inputs.append(
                        httpclient.InferInput(
                        f"INPUT{i}", input_data[i].shape, np_to_triton_dtype(input_data[i].dtype)
                        )
                    )
                for i in range(len(inputs)):
                    inputs[i].set_data_from_numpy(input_data[i])
            else:
                inputs.append(
                    httpclient.InferInput(
                    f"INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype)
                    )
                )
                inputs[0].set_data_from_numpy(input_data)

            outputs = [
                httpclient.InferRequestedOutput("OUTPUT0"),
            ]

            response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

            output0_data = response.as_numpy("OUTPUT0")
            output_queue.put(output0_data,block=False)

def while_blip_vqa_visual_encoder(input_queue,output_queue):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        while True:
            input0_data=input_queue.get()
            if input0_data is None:
                break
            inputs = [
                httpclient.InferInput(
                    "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
                ),
            ]

            inputs[0].set_data_from_numpy(input0_data)

            outputs = [
                httpclient.InferRequestedOutput("OUTPUT0"),
            ]

            response = client.infer("blip_vqa_visual_encoder", inputs, request_id=str(1), outputs=outputs)

            output0_data = response.as_numpy("OUTPUT0")
            output_queue.put(output0_data,block=False)


def while_blip_vqa_text_encoder(input_queue,output_queue):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        while True:
            input_data=input_queue.get()
            if input_data is None:
                break
            input0_data,input1_data=input_data[0],input_data[1]

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

            output0_data = response.as_numpy("OUTPUT0")
            output_queue.put(output0_data,block=False)



def while_blip_vqa_text_decoder(input_queue,output_queue):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        while True:
            input0_data = input_queue.get()
            if input0_data is None:
                break
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

            output0_data = response.as_numpy("OUTPUT0")
            output_queue.put(output0_data,block=False)


if __name__=="__main__":
    blip_vqa_visual_encoder_input_queue=queue.Queue()
    blip_vqa_visual_encoder_output_queue=queue.Queue()
    # blip_vqa_visual_encoder_thread = threading.Thread(target=while_blip_vqa_visual_encoder,args=(blip_vqa_visual_encoder_input_queue,blip_vqa_visual_encoder_output_queue))
    # blip_vqa_visual_encoder_thread.start()
    blip_vqa_visual_encoder_thread = threading.Thread(target=while_client,args=("blip_vqa_visual_encoder",blip_vqa_visual_encoder_input_queue,blip_vqa_visual_encoder_output_queue))
    blip_vqa_visual_encoder_thread.start()
    for batch_size in [2,4]:
        for _ in range(3):
            input0_data=np.array([b"/workspace/demos/images/merlion.png"]*batch_size)
            blip_vqa_visual_encoder_input_queue.put(input0_data,block=False)
    for batch_size in [2,4]:
        for _ in range(3):
            input0_data=np.array([b"/workspace/demos/images/merlion.png"]*batch_size)
            output0_data=blip_vqa_visual_encoder_output_queue.get()
            print("INPUT0 ({})  = OUTPUT0 ({})".format(input0_data.shape, output0_data.shape))

    blip_vqa_text_encoder_input_queue=queue.Queue()
    blip_vqa_text_encoder_output_queue=queue.Queue()
    # blip_vqa_text_encoder_thread = threading.Thread(target=while_blip_vqa_text_encoder,args=(blip_vqa_text_encoder_input_queue,blip_vqa_text_encoder_output_queue))
    # blip_vqa_text_encoder_thread.start()
    blip_vqa_text_encoder_thread = threading.Thread(target=while_client,args=("blip_vqa_text_encoder",blip_vqa_text_encoder_input_queue,blip_vqa_text_encoder_output_queue))
    blip_vqa_text_encoder_thread.start()
    for batch_size in [2,4]:
        for _ in range(3):
            input0_data = np.repeat(np.load("/workspace/pretrained/images_embeds.npy"),batch_size,axis=0)
            input1_data = np.array([b"where is the woman sitting?"]*batch_size)
            blip_vqa_text_encoder_input_queue.put((input0_data,input1_data),block=False)
    for batch_size in [2,4]:
        for _ in range(3):
            input0_data = np.repeat(np.load("/workspace/pretrained/images_embeds.npy"),batch_size,axis=0)
            input1_data = np.array([b"where is the woman sitting?"]*batch_size)
            output0_data=blip_vqa_text_encoder_output_queue.get()
            print(
                "INPUT0 {} + INPUT1 {} = OUTPUT0 {}".format(
                    input0_data.shape, input1_data, output0_data.shape
                )
            )
    
    blip_vqa_text_decoder_input_queue=queue.Queue()
    blip_vqa_text_decoder_output_queue=queue.Queue()
    # blip_vqa_text_decoder_thread = threading.Thread(target=while_blip_vqa_text_decoder,args=(blip_vqa_text_decoder_input_queue,blip_vqa_text_decoder_output_queue))
    # blip_vqa_text_decoder_thread.start()
    blip_vqa_text_decoder_thread = threading.Thread(target=while_client,args=("blip_vqa_text_decoder",blip_vqa_text_decoder_input_queue,blip_vqa_text_decoder_output_queue))
    blip_vqa_text_decoder_thread.start()
    for batch_size in [2,4]:
        for _ in range(3):
            input0_data = np.repeat(np.load("/workspace/pretrained/questions_states.npy"),batch_size,axis=0)
            blip_vqa_text_decoder_input_queue.put(input0_data,block=False)
    for batch_size in [2,4]:
        for _ in range(3):
            input0_data = np.repeat(np.load("/workspace/pretrained/questions_states.npy"),batch_size,axis=0)
            output0_data=blip_vqa_text_decoder_output_queue.get()
            print("INPUT0 {} = OUTPUT0 {}".format(input0_data.shape, output0_data))

    blip_vqa_visual_encoder_input_queue.put(None,block=False)
    blip_vqa_visual_encoder_thread.join()
    blip_vqa_text_encoder_input_queue.put(None,block=False)
    blip_vqa_text_encoder_thread.join()
    blip_vqa_text_decoder_input_queue.put(None,block=False)
    blip_vqa_text_decoder_thread.join()
