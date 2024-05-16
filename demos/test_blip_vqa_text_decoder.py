import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *
import torch

batch_size=3

with httpclient.InferenceServerClient("localhost:8000") as client:
    
    input0_data = np.repeat(np.load("/workspace/pretrained/questions_states.npy"),batch_size,axis=0)

    print(input0_data.shape)
    # print(input0_data.dtype)
    # (3, 1, 8, 768)
    # float32

    input1_data_shape=(input0_data.shape[0]*input0_data.shape[1],input0_data.shape[2])
    input1_data = torch.ones(input1_data_shape, dtype=torch.long).numpy(force=True)
    print(input1_data.shape)


    input0_data1 = np.random.randn(batch_size,input0_data.shape[1],input0_data.shape[2]+4,input0_data.shape[3]).astype(input0_data.dtype)
    print(input0_data1.shape)
    input1_data1_shape=(input0_data1.shape[0]*input0_data1.shape[1],input0_data1.shape[2])
    input1_data1 = torch.ones(input1_data1_shape, dtype=torch.long).numpy(force=True)
    print(input1_data1.shape)

    if input0_data.shape[2]>input0_data1.shape[2]:
        input1_data1=np.pad(input1_data1,((0,0),(input0_data.shape[2]-input0_data1.shape[2],0)),"constant", constant_values=(0, 0))
        input0_data1=np.pad(input0_data1,((0,0),(0,0),(input0_data.shape[2]-input0_data1.shape[2],0),(0,0)),"constant")
    elif input0_data.shape[2]<input0_data1.shape[2]:
        input1_data=np.pad(input1_data,((0,0),(input0_data1.shape[2]-input0_data.shape[2],0)),"constant", constant_values=(0, 0))
        input0_data=np.pad(input0_data,((0,0),(0,0),(input0_data1.shape[2]-input0_data.shape[2],0),(0,0)),"constant")


    print(input0_data.shape)
    print(input0_data1.shape)

    print(input1_data.shape)
    print(input1_data1.shape)

    input0_data=np.concatenate([input0_data,input0_data1],axis=0)
    input1_data=np.concatenate([input1_data,input1_data1],axis=0)

    print(input0_data.shape)
    print(input1_data)

    
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
        "blip_vqa_text_decoder", inputs, request_id=str(1), outputs=outputs
    )

    output0_data = response.as_numpy("OUTPUT0")

    print("INPUT0 {} INPUT1 {} = OUTPUT0 {}".format(input0_data.shape,input1_data.shape,output0_data))