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

def blip_vqa_visual_encoder_task(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue):
    blip_vqa_visual_encoder_input_queue=queue.Queue()
    blip_vqa_visual_encoder_output_queue=queue.Queue()
    blip_vqa_visual_encoder_thread = threading.Thread(target=while_client,args=("blip_vqa_visual_encoder",blip_vqa_visual_encoder_input_queue,blip_vqa_visual_encoder_output_queue))
    blip_vqa_visual_encoder_thread.start()
    while True:
        blip_vqa_visual_encoder_batches=blip_vqa_visual_encoder_batches_queue.get()
        if blip_vqa_visual_encoder_batches is None:
            break
        for images_per_batch in blip_vqa_visual_encoder_batches:
            blip_vqa_visual_encoder_input_queue.put(images_per_batch,block=False)
        for images_per_batch in blip_vqa_visual_encoder_batches:
            blip_vqa_visual_encoder_blip_vqa_text_encoder_queue.put(blip_vqa_visual_encoder_output_queue.get(),block=False)
    blip_vqa_visual_encoder_input_queue.put(None,block=False)
    blip_vqa_visual_encoder_thread.join()

def blip_vqa_text_encoder_task(blip_vqa_text_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue):
    blip_vqa_text_encoder_input_queue=queue.Queue()
    blip_vqa_text_encoder_output_queue=queue.Queue()
    blip_vqa_text_encoder_thread = threading.Thread(target=while_client,args=("blip_vqa_text_encoder",blip_vqa_text_encoder_input_queue,blip_vqa_text_encoder_output_queue))
    blip_vqa_text_encoder_thread.start()
    while True:
        blip_vqa_text_encoder_batches=blip_vqa_text_encoder_batches_queue.get()
        if blip_vqa_text_encoder_batches is None:
            break
        text_batch_num=len(blip_vqa_text_encoder_batches)
        # print(text_batch_num)
        text_batch_input_count=0
        text_batch_output_count=0
        now_left=None
        while text_batch_input_count<text_batch_num or text_batch_output_count<text_batch_num:
            # cache pad image replication
            try:
                images_per_batch=blip_vqa_visual_encoder_blip_vqa_text_encoder_queue.get(block=False)
                if now_left is None:
                    now_left=images_per_batch
                else:
                    now_left=np.concatenate([now_left,images_per_batch],axis=0)
                # print(text_batch_input_count,now_left.shape[0],blip_vqa_text_encoder_batches[text_batch_input_count].shape[0])
            except queue.Empty:
                pass

            while now_left is not None and text_batch_input_count<text_batch_num and now_left.shape[0]>=blip_vqa_text_encoder_batches[text_batch_input_count].shape[0]:
                # print(blip_vqa_text_encoder_batches[text_batch_input_count])
                blip_vqa_text_encoder_input_queue.put((now_left[:blip_vqa_text_encoder_batches[text_batch_input_count].shape[0]],blip_vqa_text_encoder_batches[text_batch_input_count]),block=False)
                now_left=now_left[blip_vqa_text_encoder_batches[text_batch_input_count].shape[0]:]
                text_batch_input_count+=1

            try:
                texts_per_batch=blip_vqa_text_encoder_output_queue.get(block=False)
                text_batch_output_count+=1
                blip_vqa_text_encoder_blip_vqa_text_decoder_queue.put(texts_per_batch,block=False)
            except queue.Empty:
                pass      
    blip_vqa_text_encoder_input_queue.put(None,block=False)
    blip_vqa_text_encoder_thread.join()


def blip_vqa_text_decoder_task(blip_vqa_text_decoder_batches_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,blip_vqa_text_decoder_batches_return_queue):
    blip_vqa_text_decoder_input_queue=queue.Queue()
    blip_vqa_text_decoder_output_queue=queue.Queue()
    blip_vqa_text_decoder_thread = threading.Thread(target=while_client,args=("blip_vqa_text_decoder",blip_vqa_text_decoder_input_queue,blip_vqa_text_decoder_output_queue))
    blip_vqa_text_decoder_thread.start()
    while True:
        blip_vqa_text_decoder_batches=blip_vqa_text_decoder_batches_queue.get()
        if blip_vqa_text_decoder_batches is None:
            break
        text_batch_num=len(blip_vqa_text_decoder_batches)
        text_batch_input_count=0
        text_batch_output_count=0
        now_left=None
        while text_batch_input_count<text_batch_num or text_batch_output_count<text_batch_num:
            # cache pad image replication
            try:
                texts_per_batch=blip_vqa_text_encoder_blip_vqa_text_decoder_queue.get(block=False)
                if now_left is None:
                    now_left=texts_per_batch
                else:
                    now_left=np.concatenate([now_left,texts_per_batch],axis=0)
                # print(text_batch_input_count,now_left.shape[0],blip_vqa_text_decoder_batches[text_batch_input_count].shape[0])
            except queue.Empty:
                pass

            while now_left is not None and text_batch_input_count<text_batch_num and now_left.shape[0]>=blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]:
                # print(blip_vqa_text_decoder_batches[text_batch_input_count])
                blip_vqa_text_decoder_input_queue.put(now_left[:blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]],block=False)
                now_left=now_left[blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]:]
                text_batch_input_count+=1
            
            try:
                texts_per_batch=blip_vqa_text_decoder_output_queue.get(block=False)
                text_batch_output_count+=1
                blip_vqa_text_decoder_batches_return_queue.put(texts_per_batch,block=False)
            except queue.Empty:
                pass      
    blip_vqa_text_decoder_input_queue.put(None,block=False)  
    blip_vqa_text_decoder_thread.join()

if __name__=="__main__":

    blip_vqa_visual_encoder_blip_vqa_text_encoder_queue=queue.Queue()
    blip_vqa_text_encoder_blip_vqa_text_decoder_queue=queue.Queue()

    blip_vqa_visual_encoder_batches_queue=queue.Queue()
    blip_vqa_visual_encoder_task_thread=threading.Thread(target=blip_vqa_visual_encoder_task,args=(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue))
    blip_vqa_visual_encoder_task_thread.start()

    blip_vqa_text_encoder_batches_queue=queue.Queue()
    blip_vqa_text_encoder_task_thread=threading.Thread(target=blip_vqa_text_encoder_task,args=(blip_vqa_text_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue))
    blip_vqa_text_encoder_task_thread.start()

    blip_vqa_text_decoder_batches_queue=queue.Queue()
    blip_vqa_text_decoder_batches_return_queue=queue.Queue()
    blip_vqa_text_decoder_task_thread=threading.Thread(target=blip_vqa_text_decoder_task,args=(blip_vqa_text_decoder_batches_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,blip_vqa_text_decoder_batches_return_queue))
    blip_vqa_text_decoder_task_thread.start()

    blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=15,15,15
    print(blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size)

    batch_size=15
    images=np.array([b"/workspace/demos/images/merlion.png"]*batch_size)
    texts=np.array([b"where is the woman sitting?"]*batch_size)

    if images.shape[0]<=blip_vqa_visual_encoder_batch_size:
        blip_vqa_visual_encoder_batches=[images]
        num_blip_vqa_visual_encoder_batch=1
    else:
        num_blip_vqa_visual_encoder_batch=images.shape[0]//blip_vqa_visual_encoder_batch_size
        num_blip_vqa_visual_encoder_left=images.shape[0]-num_blip_vqa_visual_encoder_batch*blip_vqa_visual_encoder_batch_size
        if num_blip_vqa_visual_encoder_left!=0:
            blip_vqa_visual_encoder_batches=np.split(images[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[images[-num_blip_vqa_visual_encoder_left:]]
            num_blip_vqa_visual_encoder_batch+=1
        else:
            blip_vqa_visual_encoder_batches=np.split(images,num_blip_vqa_visual_encoder_batch)

    if texts.shape[0]<=blip_vqa_text_encoder_batch_size:
        blip_vqa_text_encoder_batches=[texts]
        num_blip_vqa_text_encoder_batch=1
    else:
        num_blip_vqa_text_encoder_batch=texts.shape[0]//blip_vqa_text_encoder_batch_size
        num_blip_vqa_text_encoder_left=texts.shape[0]-num_blip_vqa_text_encoder_batch*blip_vqa_text_encoder_batch_size
        if num_blip_vqa_text_encoder_left!=0:
            blip_vqa_text_encoder_batches=np.split(texts[:-num_blip_vqa_text_encoder_left],num_blip_vqa_text_encoder_batch)+[texts[-num_blip_vqa_text_encoder_left:]]
            num_blip_vqa_text_encoder_batch+=1
        else:
            blip_vqa_text_encoder_batches=np.split(texts,num_blip_vqa_text_encoder_batch)
    
    if texts.shape[0]<=blip_vqa_text_decoder_batch_size:
        blip_vqa_text_decoder_batches=[texts]
        num_blip_vqa_text_decoder_batch=1
    else:
        num_blip_vqa_text_decoder_batch=texts.shape[0]//blip_vqa_text_decoder_batch_size
        num_blip_vqa_text_decoder_left=texts.shape[0]-num_blip_vqa_text_decoder_batch*blip_vqa_text_decoder_batch_size
        if num_blip_vqa_text_decoder_left!=0:
            blip_vqa_text_decoder_batches=np.split(texts[:-num_blip_vqa_text_decoder_left],num_blip_vqa_text_decoder_batch)+[texts[-num_blip_vqa_text_decoder_left:]]
            num_blip_vqa_text_decoder_batch+=1
        else:
            blip_vqa_text_decoder_batches=np.split(texts,num_blip_vqa_text_decoder_batch)

    '''
    print(len(images))
    print(len(texts))
    print(len(blip_vqa_visual_encoder_batches))
    for i in blip_vqa_visual_encoder_batches:
        print(len(i))
    print(len(blip_vqa_text_encoder_batches))
    for i in blip_vqa_text_encoder_batches:
        print(len(i))
    print(len(blip_vqa_text_decoder_batches))
    for i in blip_vqa_text_decoder_batches:
        print(len(i))
    '''

    blip_vqa_visual_encoder_batches_queue.put(blip_vqa_visual_encoder_batches,block=False)

    # for _ in range(num_blip_vqa_visual_encoder_batch):
    #     blip_vqa_visual_encoder_batches_return=blip_vqa_visual_encoder_blip_vqa_text_encoder_queue.get()
    #     print(blip_vqa_visual_encoder_batches_return.shape)
    # blip_vqa_visual_encoder_batches_queue.put(None,block=False)
    # blip_vqa_visual_encoder_task_thread.join()
    
    blip_vqa_text_encoder_batches_queue.put(blip_vqa_text_encoder_batches,block=False)
    
    # for _ in range(num_blip_vqa_text_encoder_batch):
    #     blip_vqa_text_encoder_batches_return=blip_vqa_text_encoder_blip_vqa_text_decoder_queue.get()
    #     print(blip_vqa_text_encoder_batches_return.shape)
    # blip_vqa_visual_encoder_batches_queue.put(None,block=False)
    # blip_vqa_visual_encoder_task_thread.join()
    # blip_vqa_text_encoder_batches_queue.put(None,block=False)
    # blip_vqa_text_encoder_task_thread.join()

    blip_vqa_text_decoder_batches_queue.put(blip_vqa_text_decoder_batches,block=False)

    batch_count=0
    now_left=None
    for _ in range(num_blip_vqa_text_decoder_batch):
        blip_vqa_text_decoder_batches_return=blip_vqa_text_decoder_batches_return_queue.get()
        print(blip_vqa_text_decoder_batches_return.shape)
        # print(blip_vqa_text_decoder_batches_return)

    blip_vqa_visual_encoder_batches_queue.put(None,block=False)
    blip_vqa_visual_encoder_task_thread.join()
    blip_vqa_text_encoder_batches_queue.put(None,block=False)
    blip_vqa_text_encoder_task_thread.join()
    blip_vqa_text_decoder_batches_queue.put(None,block=False)
    blip_vqa_text_decoder_task_thread.join()