import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import tritonclient.http as httpclient
from tritonclient.utils import *
import time
import multiprocessing
import queue
import random
import os
from cache import cache_get_put,recover_cache_duplication,recover_once

def while_client(model_name,input_queue,output_queue):
    try:
        with httpclient.InferenceServerClient("localhost:8000") as client:
            while True:
                input_data=input_queue.get()
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
    except KeyboardInterrupt:
        pass

def blip_vqa_visual_encoder_task(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,cache_put_queue):
    try:
        blip_vqa_visual_encoder_input_queue=multiprocessing.Queue()
        blip_vqa_visual_encoder_output_queue=multiprocessing.Queue()
        processes=[]
        blip_vqa_visual_encoder_process = multiprocessing.Process(target=while_client,args=("blip_vqa_visual_encoder",blip_vqa_visual_encoder_input_queue,blip_vqa_visual_encoder_output_queue))
        processes.append(blip_vqa_visual_encoder_process)
        blip_vqa_visual_encoder_process.start()

        while True:
            batches_queue_output=blip_vqa_visual_encoder_batches_queue.get()
            if isinstance(batches_queue_output,tuple):
                images,unique_images,cached_images_result,no_forward_indices,blip_vqa_visual_encoder_batches,forward_indices_batches=batches_queue_output
                # last two are splited   
            else:
                blip_vqa_visual_encoder_batches=batches_queue_output

            if len(blip_vqa_visual_encoder_batches)==0:
                # all cached
                recover_images_from_duplication=recover_once(images,unique_images,cached_images_result)
                blip_vqa_visual_encoder_blip_vqa_text_encoder_queue.put(recover_images_from_duplication)
                continue

            for images_per_batch in blip_vqa_visual_encoder_batches:
                blip_vqa_visual_encoder_input_queue.put(images_per_batch,block=False)

            recover_images_from_duplication=None
            mask=None
            no_forward_indices_indice_start=0
            start_to_send_indice=0
            for i,images_per_batch in enumerate(blip_vqa_visual_encoder_batches):
                output_per_batch=blip_vqa_visual_encoder_output_queue.get()
                if isinstance(batches_queue_output,tuple):
                    print("images_per_batch")
                    print(images_per_batch)
                    print("output_per_batch")
                    print(output_per_batch.shape)
                    # add cache
                    cache_put_queue.put((images_per_batch,output_per_batch),block=False)
                    
                    print("np.all((no_forward_indices>forward_indices_batches[i][-1])==False)")
                    print(np.all((no_forward_indices>forward_indices_batches[i][-1])==False))
                    if np.all((no_forward_indices>forward_indices_batches[i][-1])==False):
                        # no_forward_indices all smaller than the last element of forward_indices_batches[i]
                        # all cached can be forward in this time
                        no_forward_indices_indice=no_forward_indices.shape[0]-1       
                    else:
                        no_forward_indices_indice=np.argmax(no_forward_indices>forward_indices_batches[i][-1])
                        # find the first element of no_forward_indices bigger than the last element of forward_indices_batches[i]
                        # the last indice of cached can be forward in this time
                    if recover_images_from_duplication is None:
                        recover_images_from_duplication=np.empty((images.shape[0],*(output_per_batch.shape[1:])),dtype=output_per_batch.dtype)
                        mask=np.full(images.shape[0],False)

                    recover_images_from_duplication,mask=recover_cache_duplication(images,unique_images,cached_images_result[no_forward_indices_indice_start:no_forward_indices_indice+1],no_forward_indices[no_forward_indices_indice_start:no_forward_indices_indice+1],output_per_batch,forward_indices_batches[i],recover_images_from_duplication,mask)
                    no_forward_indices_indice_start=no_forward_indices_indice+1
                    print(mask)

                    # send as you can
                    inc=0
                    while start_to_send_indice<images.shape[0] and start_to_send_indice+inc<images.shape[0] and np.all(mask[start_to_send_indice:start_to_send_indice+inc+1] == True):
                        inc+=1
                    print("inc")
                    print(inc)
                    blip_vqa_visual_encoder_blip_vqa_text_encoder_queue.put(recover_images_from_duplication[start_to_send_indice:start_to_send_indice+inc],block=False)
                    start_to_send_indice+=inc
                else:
                    blip_vqa_visual_encoder_blip_vqa_text_encoder_queue.put(output_per_batch,block=False)
    except KeyboardInterrupt:
        for process in processes:
            process.join()

def blip_vqa_text_encoder_task(blip_vqa_text_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue):
    try:
        blip_vqa_text_encoder_input_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_output_queue=multiprocessing.Queue()
        processes=[]
        blip_vqa_text_encoder_process = multiprocessing.Process(target=while_client,args=("blip_vqa_text_encoder",blip_vqa_text_encoder_input_queue,blip_vqa_text_encoder_output_queue))
        processes.append(blip_vqa_text_encoder_process)
        blip_vqa_text_encoder_process.start()
        while True:
            blip_vqa_text_encoder_batches=blip_vqa_text_encoder_batches_queue.get()
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
    except KeyboardInterrupt:
        for process in processes:
            process.join()


def pad_concate(input0_data,input1_data,input0_data1):
    input1_data1_shape=(input0_data1.shape[0]*input0_data1.shape[1],input0_data1.shape[2])
    input1_data1 = torch.ones(input1_data1_shape, dtype=torch.long).numpy(force=True)

    if input0_data.shape[2]>input0_data1.shape[2]:
        input1_data1=np.pad(input1_data1,((0,0),(input0_data.shape[2]-input0_data1.shape[2],0)),"constant", constant_values=(0, 0))
        input0_data1=np.pad(input0_data1,((0,0),(0,0),(input0_data.shape[2]-input0_data1.shape[2],0),(0,0)),"constant")
    elif input0_data.shape[2]<input0_data1.shape[2]:
        input1_data=np.pad(input1_data,((0,0),(input0_data1.shape[2]-input0_data.shape[2],0)),"constant", constant_values=(0, 0))
        input0_data=np.pad(input0_data,((0,0),(0,0),(input0_data1.shape[2]-input0_data.shape[2],0),(0,0)),"constant")

    input0_data=np.concatenate([input0_data,input0_data1],axis=0)
    input1_data=np.concatenate([input1_data,input1_data1],axis=0)
    return input0_data,input1_data

def blip_vqa_text_decoder_task(blip_vqa_text_decoder_batches_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,blip_vqa_text_decoder_batches_return_queue):
    try:
        blip_vqa_text_decoder_input_queue=multiprocessing.Queue()
        blip_vqa_text_decoder_output_queue=multiprocessing.Queue()
        processes=[]
        blip_vqa_text_decoder_process = multiprocessing.Process(target=while_client,args=("blip_vqa_text_decoder",blip_vqa_text_decoder_input_queue,blip_vqa_text_decoder_output_queue))
        processes.append(blip_vqa_text_decoder_process)
        blip_vqa_text_decoder_process.start()
        while True:
            blip_vqa_text_decoder_batches=blip_vqa_text_decoder_batches_queue.get()
            text_batch_num=len(blip_vqa_text_decoder_batches)
            text_batch_input_count=0
            text_batch_output_count=0
            now_left_data,now_left_mask=None,None
            while text_batch_input_count<text_batch_num or text_batch_output_count<text_batch_num:
                try:
                    texts_per_batch=blip_vqa_text_encoder_blip_vqa_text_decoder_queue.get(block=False)
                    if now_left_data is None:
                        now_left_data=texts_per_batch
                        now_left_mask_shape=(now_left_data.shape[0]*now_left_data.shape[1],now_left_data.shape[2])
                        now_left_mask = torch.ones(now_left_mask_shape, dtype=torch.long).numpy(force=True)
                    else:                        
                        now_left_data,now_left_mask=pad_concate(now_left_data,now_left_mask,texts_per_batch)
                    # print(text_batch_input_count,now_left.shape[0],blip_vqa_text_decoder_batches[text_batch_input_count].shape[0])
                except queue.Empty:
                    pass

                while now_left_data is not None and text_batch_input_count<text_batch_num and now_left_data.shape[0]>=blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]:
                    # print(blip_vqa_text_decoder_batches[text_batch_input_count])
                    blip_vqa_text_decoder_input_queue.put((now_left_data[:blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]],now_left_mask[:blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]]),block=False)
                    now_left_data=now_left_data[blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]:]
                    now_left_mask=now_left_mask[blip_vqa_text_decoder_batches[text_batch_input_count].shape[0]:]
                    text_batch_input_count+=1
                
                try:
                    texts_per_batch=blip_vqa_text_decoder_output_queue.get(block=False)
                    text_batch_output_count+=1
                    blip_vqa_text_decoder_batches_return_queue.put(texts_per_batch,block=False)
                except queue.Empty:
                    pass       
    except KeyboardInterrupt:
        for process in processes:
            process.join()

def change_batch_size(batch_size_queue,time_interval):
    try:
        while True:
            a,b,c=random.randint(1, 10),random.randint(1, 10),random.randint(1, 10)
            batch_size_queue.put((a,b,c),block=False)
            # print(f"release batch size: {a},{b},{c}")
            time.sleep(time_interval)
    except KeyboardInterrupt:
        pass

def blip_vqa_process(request_queue,request_events,processed_results,batch_size_queue):
    try:
        # write_file="/workspace/blip_vqa_process.txt"
        # if(os.path.isfile(write_file)):    
        #     os.remove(write_file)

        processes = []

        cache_get_input_queue=multiprocessing.Queue()
        cache_get_output_queue=multiprocessing.Queue()
        cache_put_queue=multiprocessing.Queue()
        cache_process=multiprocessing.Process(target=cache_get_put,args=(cache_get_input_queue,cache_get_output_queue,cache_put_queue,))
        processes.append(cache_process)
        cache_process.start()

        blip_vqa_visual_encoder_blip_vqa_text_encoder_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_blip_vqa_text_decoder_queue=multiprocessing.Queue()

        blip_vqa_visual_encoder_batches_queue=multiprocessing.Queue()
        blip_vqa_visual_encoder_task_process=multiprocessing.Process(target=blip_vqa_visual_encoder_task,args=(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,cache_put_queue))
        processes.append(blip_vqa_visual_encoder_task_process)
        blip_vqa_visual_encoder_task_process.start()

        blip_vqa_text_encoder_batches_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_task_process=multiprocessing.Process(target=blip_vqa_text_encoder_task,args=(blip_vqa_text_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,))
        processes.append(blip_vqa_text_encoder_task_process)
        blip_vqa_text_encoder_task_process.start()

        blip_vqa_text_decoder_batches_queue=multiprocessing.Queue()
        blip_vqa_text_decoder_batches_return_queue=multiprocessing.Queue()
        blip_vqa_text_decoder_task_process=multiprocessing.Process(target=blip_vqa_text_decoder_task,args=(blip_vqa_text_decoder_batches_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,blip_vqa_text_decoder_batches_return_queue,))
        processes.append(blip_vqa_text_decoder_task_process)
        blip_vqa_text_decoder_task_process.start()

        blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=1,1,1
        
        while True:
            try:
                batch_sizes=batch_size_queue.get(block=False)
                blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=batch_sizes[0],batch_sizes[1],batch_sizes[2]
                print(f"update batch sizes: {batch_sizes}")
            except queue.Empty:
                pass
            if not request_queue.empty():
                request_ids, batch_nums,images, texts=[],[],[],[]
                while not request_queue.empty():
                    request_id,image_batch,text_batch=request_queue.get()
                    request_ids.append(request_id)
                    images.append(image_batch)
                    texts.append(text_batch)
                    batch_nums.append(image_batch.shape[0])
                print(f"request num: {len(request_ids)}")
                
                images=np.concatenate(images, axis=0)
                texts=np.concatenate(texts, axis=0)

                cache_get_input_queue.put(images,block=False)
                unique_images,cached_images_result,no_forward_indices,unique_images_forward,forward_indices=cache_get_output_queue.get()
                #all np
                
                if unique_images_forward.shape[0]==0:
                    blip_vqa_visual_encoder_batches=[]                    
                    forward_indices_batches=[]
                    num_blip_vqa_visual_encoder_batch=0
                elif unique_images_forward.shape[0]<=blip_vqa_visual_encoder_batch_size:
                    blip_vqa_visual_encoder_batches=[unique_images_forward]                    
                    forward_indices_batches=[forward_indices]
                    num_blip_vqa_visual_encoder_batch=1
                else:
                    num_blip_vqa_visual_encoder_batch=unique_images_forward.shape[0]//blip_vqa_visual_encoder_batch_size
                    num_blip_vqa_visual_encoder_left=unique_images_forward.shape[0]-num_blip_vqa_visual_encoder_batch*blip_vqa_visual_encoder_batch_size
                    if num_blip_vqa_visual_encoder_left!=0:
                        blip_vqa_visual_encoder_batches=np.split(unique_images_forward[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[unique_images_forward[-num_blip_vqa_visual_encoder_left:]]
                        forward_indices_batches=np.split(forward_indices[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[forward_indices[-num_blip_vqa_visual_encoder_left:]]
                        num_blip_vqa_visual_encoder_batch+=1
                    else:
                        blip_vqa_visual_encoder_batches=np.split(unique_images_forward,num_blip_vqa_visual_encoder_batch)
                        forward_indices_batches=np.split(forward_indices,num_blip_vqa_visual_encoder_batch)

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

                blip_vqa_visual_encoder_batches_queue.put((images,unique_images,cached_images_result,no_forward_indices,blip_vqa_visual_encoder_batches,forward_indices_batches),block=False)            
                blip_vqa_text_encoder_batches_queue.put(blip_vqa_text_encoder_batches,block=False)
                blip_vqa_text_decoder_batches_queue.put(blip_vqa_text_decoder_batches,block=False)

                batch_count=0
                now_left=None
                for _ in range(num_blip_vqa_text_decoder_batch):
                    blip_vqa_text_decoder_batches_return=blip_vqa_text_decoder_batches_return_queue.get()
                    if now_left is None:
                        now_left=blip_vqa_text_decoder_batches_return
                    else:
                        now_left=np.concatenate([now_left,blip_vqa_text_decoder_batches_return],axis=0)
                    while batch_count<len(batch_nums) and now_left.shape[0]>=batch_nums[batch_count]:
                        post_return=now_left[:batch_nums[batch_count]]
                        now_left=now_left[batch_nums[batch_count]:]
                        processed_results[request_ids[batch_count]] = post_return
                        request_events[request_ids[batch_count]]=1
                        print(f"one request finish, batch size: {batch_nums[batch_count]}, request id: {batch_count}")
                        batch_count+=1      
    except KeyboardInterrupt:
        for process in processes:
            process.join()

if __name__=="__main__":

    try:
        # Create a list to hold process objects
        processes = []

        cache_get_input_queue=multiprocessing.Queue()
        cache_get_output_queue=multiprocessing.Queue()
        cache_put_queue=multiprocessing.Queue()
        cache_process=multiprocessing.Process(target=cache_get_put,args=(cache_get_input_queue,cache_get_output_queue,cache_put_queue,))
        processes.append(cache_process)
        cache_process.start()

        blip_vqa_visual_encoder_blip_vqa_text_encoder_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_blip_vqa_text_decoder_queue=multiprocessing.Queue()

        blip_vqa_visual_encoder_batches_queue=multiprocessing.Queue()
        blip_vqa_visual_encoder_task_process=multiprocessing.Process(target=blip_vqa_visual_encoder_task,args=(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,cache_put_queue))
        processes.append(blip_vqa_visual_encoder_task_process)
        blip_vqa_visual_encoder_task_process.start()

        blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=2,2,2

        for _ in range(2):
            batch_size=5*2
            images = [
                "/workspace/demos/images/merlion.png","/workspace/demos/images/beach.jpg"
            ]*(batch_size//2)
            texts = [
                "where is it?","where is the woman sitting?"
            ]*(batch_size//2)
            images=np.array([image.encode('utf-8') for image in images])
            texts=np.array([text.encode('utf-8') for text in texts])

            cache_get_input_queue.put(images,block=False)
            unique_images,cached_images_result,no_forward_indices,unique_images_forward,forward_indices=cache_get_output_queue.get() 
            
            if unique_images_forward.shape[0]==0:
                blip_vqa_visual_encoder_batches=[]                    
                forward_indices_batches=[]
                num_blip_vqa_visual_encoder_batch=0
            elif unique_images_forward.shape[0]<=blip_vqa_visual_encoder_batch_size:
                blip_vqa_visual_encoder_batches=[unique_images_forward]                    
                forward_indices_batches=[forward_indices]
                num_blip_vqa_visual_encoder_batch=1
            else:
                num_blip_vqa_visual_encoder_batch=unique_images_forward.shape[0]//blip_vqa_visual_encoder_batch_size
                num_blip_vqa_visual_encoder_left=unique_images_forward.shape[0]-num_blip_vqa_visual_encoder_batch*blip_vqa_visual_encoder_batch_size
                if num_blip_vqa_visual_encoder_left!=0:
                    blip_vqa_visual_encoder_batches=np.split(unique_images_forward[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[unique_images_forward[-num_blip_vqa_visual_encoder_left:]]
                    forward_indices_batches=np.split(forward_indices[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[forward_indices[-num_blip_vqa_visual_encoder_left:]]
                    num_blip_vqa_visual_encoder_batch+=1
                else:
                    blip_vqa_visual_encoder_batches=np.split(unique_images_forward,num_blip_vqa_visual_encoder_batch)
                    forward_indices_batches=np.split(forward_indices,num_blip_vqa_visual_encoder_batch)
                
            print("images")
            print(images)
            print("unique_images")
            print(unique_images)
            print("cached_images_result")
            print(cached_images_result.shape)
            print("no_forward_indices")
            print(no_forward_indices)
            print("blip_vqa_visual_encoder_batches")
            print(blip_vqa_visual_encoder_batches)
            print("forward_indices_batches")
            print(forward_indices_batches)

            blip_vqa_visual_encoder_batches_queue.put((images,unique_images,cached_images_result,no_forward_indices,blip_vqa_visual_encoder_batches,forward_indices_batches),block=False)            

            has_output_num=0
            while True:
                output_per_batch=blip_vqa_visual_encoder_blip_vqa_text_encoder_queue.get()
                print("output_per_batch.shape")
                print(output_per_batch.shape)
                has_output_num+=output_per_batch.shape[0]
                print(f"has_output_num: {has_output_num}")
                if has_output_num==images.shape[0]:
                    break
            print("------------------------------------------------")
        
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("finish...")

