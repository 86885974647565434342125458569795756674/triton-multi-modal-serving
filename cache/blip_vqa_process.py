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
from cache import LRUCache,remove_duplication_cache,recover_cache_duplication

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

def blip_vqa_visual_encoder_task(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue):
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
                images_batches,unique_images_batches,cached_images_result_batches,no_forward_indices_batches,blip_vqa_visual_encoder_batches,forward_indices_batches=batches_queue_output[0],batches_queue_output[1],batches_queue_output[2],batches_queue_output[3],batches_queue_output[4],batches_queue_output[5]
            else:
                blip_vqa_visual_encoder_batches=batches_queue_output
            for images_per_batch in blip_vqa_visual_encoder_batches:
                blip_vqa_visual_encoder_input_queue.put(images_per_batch,block=False)
            for i,images_per_batch in enumerate(blip_vqa_visual_encoder_batches):
                output_per_batch=blip_vqa_visual_encoder_output_queue.get()
                if isinstance(batches_queue_output,tuple):
                    _,recover_images_from_duplication=recover_cache_duplication(images_batches[i],unique_images_batches[i],cached_images_result_batches[i],no_forward_indices_batches[i],output_per_batch,forward_indices_batches[i])
                    output_per_batch=recover_images_from_duplication
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

def blip_vqa_process_queue(request_queue,request_events,processed_results,batch_size_queue):
    try:
        # write_file="/workspace/blip_vqa_process_queue.txt"
        # if(os.path.isfile(write_file)):    
        #     os.remove(write_file)
        capacity=100
        image_cache=LRUCache(capacity)

        blip_vqa_visual_encoder_blip_vqa_text_encoder_queue=multiprocessing.Queue()
        blip_vqa_text_encoder_blip_vqa_text_decoder_queue=multiprocessing.Queue()

        processes = []
        blip_vqa_visual_encoder_batches_queue=multiprocessing.Queue()
        blip_vqa_visual_encoder_task_process=multiprocessing.Process(target=blip_vqa_visual_encoder_task,args=(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,))
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
                    post_data = request_queue.get()
                    request_id,image,text=post_data[0],post_data[1],post_data[2]
                    request_ids.append(request_id)
                    images.append(image)
                    texts.append(text)
                    batch_nums.append(image.shape[0])
                print(f"request num: {len(request_ids)}")
                
                images=np.concatenate(images, axis=0)
                texts=np.concatenate(texts, axis=0)

                unique_images,cached_images_result,no_forward_indices,unique_images_forward,forward_indices=remove_duplication_cache(image_cache,images)
                
                indices=np.array([i for i in range(unique_images_forward.shape[0])])
                if unique_images_forward.shape[0]<=blip_vqa_visual_encoder_batch_size:
                    blip_vqa_visual_encoder_batches=[unique_images_forward]
                    
                    cached_images_result_batches=[cached_images_result]
                    no_forward_indices_batches=[no_forward_indices_batches]
                    forward_indices_batches=[forward_indices_batches]
                    images_batches=[images]
                    unique_images_batches=[unique_images]
                    
                    num_blip_vqa_visual_encoder_batch=1
                else:
                    num_blip_vqa_visual_encoder_batch=unique_images_forward.shape[0]//blip_vqa_visual_encoder_batch_size
                    num_blip_vqa_visual_encoder_left=unique_images_forward.shape[0]-num_blip_vqa_visual_encoder_batch*blip_vqa_visual_encoder_batch_size
                    if num_blip_vqa_visual_encoder_left!=0:
                        blip_vqa_visual_encoder_batches=np.split(unique_images_forward[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[unique_images_forward[-num_blip_vqa_visual_encoder_left:]]
                        
                        indices_batches=np.split(indices[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[indices[-num_blip_vqa_visual_encoder_left:]]
                        no_forward_indices_batches=[]
                        cached_images_result_batches=[]
                        start_cached=0
                        for indices_batch in indices_batches:
                            mask=np.isin(indices_batch,no_forward_indices)
                            no_forward_indices_batches.append(indices_batch[mask])
                            cached_images_result_batches.append(cached_images_result[start_cached:start_cached+len(indices_batch[mask])])
                            start_cached+=len(indices_batch[mask])
                        forward_indices_batches=[]
                        for indices_batch in indices_batches:
                            mask=np.isin(indices_batch,forward_indices)
                            forward_indices_batches.append(indices_batch[mask])
                            
                        images_batches=np.split(images[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[images[-num_blip_vqa_visual_encoder_left:]]
                        unique_images_batches=np.split(unique_images[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)+[unique_images[-num_blip_vqa_visual_encoder_left:]]

                        num_blip_vqa_visual_encoder_batch+=1
                    else:
                        blip_vqa_visual_encoder_batches=np.split(unique_images_forward,num_blip_vqa_visual_encoder_batch)
                        
                        indices_batches=np.split(indices[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)
                        no_forward_indices_batches=[]
                        cached_images_result_batches=[]
                        start_cached=0
                        for indices_batch in indices_batches:
                            mask=np.isin(indices_batch,no_forward_indices)
                            no_forward_indices_batches.append(indices_batch[mask])
                            cached_images_result_batches.append(cached_images_result[start_cached:start_cached+len(indices_batch[mask])])
                            start_cached+=len(indices_batch[mask])
                        forward_indices_batches=[]
                        for indices_batch in indices_batches:
                            mask=np.isin(indices_batch,forward_indices)
                            forward_indices_batches.append(indices_batch[mask])

                        images_batches=np.split(images[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)
                        unique_images_batches=np.split(unique_images[:-num_blip_vqa_visual_encoder_left],num_blip_vqa_visual_encoder_batch)


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

                blip_vqa_visual_encoder_batches_queue.put((images_batches,unique_images_batches,cached_images_result_batches,no_forward_indices_batches,blip_vqa_visual_encoder_batches,forward_indices_batches),block=False)            
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
        batch_size_queue=multiprocessing.Queue()
        time_interval=1
        
        request_queue=multiprocessing.Queue()
        manager = multiprocessing.Manager()
        request_events=manager.dict()
        processed_results = manager.dict()

        # Create a list to hold process objects
        processes = []

        change_batch_size_process = multiprocessing.Process(target=change_batch_size, args=(batch_size_queue,time_interval))
        processes.append(change_batch_size_process)
        change_batch_size_process.start()

        blip_vqa_process_queue_process = multiprocessing.Process(target=blip_vqa_process_queue, args=(request_queue,request_events,processed_results,batch_size_queue,))
        processes.append(blip_vqa_process_queue_process)
        blip_vqa_process_queue_process.start()

        batch_size=20
        images = [
            "/workspace/demos/images/merlion.png"
        ]*batch_size
        texts = [
            "where is the woman sitting?"
        ]*batch_size
        images=np.array([image.encode('utf-8') for image in images])
        texts=np.array([text.encode('utf-8') for text in texts])
        request_id=time.time()

        # Put the request in the queue
        request_queue.put((request_id,images,texts))

        request_events[request_id]=0

        while request_events[request_id]==0:
            pass

        del request_events[request_id]

        result = processed_results.get(request_id, b"Post data not available")
        
        
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()

    exit(0)

    blip_vqa_visual_encoder_blip_vqa_text_encoder_queue=multiprocessing.Queue()
    blip_vqa_text_encoder_blip_vqa_text_decoder_queue=multiprocessing.Queue()

    blip_vqa_visual_encoder_batches_queue=multiprocessing.Queue()
    blip_vqa_visual_encoder_task_process=multiprocessing.Process(target=blip_vqa_visual_encoder_task,args=(blip_vqa_visual_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue))
    blip_vqa_visual_encoder_task_process.start()

    blip_vqa_text_encoder_batches_queue=multiprocessing.Queue()
    blip_vqa_text_encoder_task_process=multiprocessing.Process(target=blip_vqa_text_encoder_task,args=(blip_vqa_text_encoder_batches_queue,blip_vqa_visual_encoder_blip_vqa_text_encoder_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue))
    blip_vqa_text_encoder_task_process.start()

    blip_vqa_text_decoder_batches_queue=multiprocessing.Queue()
    blip_vqa_text_decoder_batches_return_queue=multiprocessing.Queue()
    blip_vqa_text_decoder_task_process=multiprocessing.Process(target=blip_vqa_text_decoder_task,args=(blip_vqa_text_decoder_batches_queue,blip_vqa_text_encoder_blip_vqa_text_decoder_queue,blip_vqa_text_decoder_batches_return_queue))
    blip_vqa_text_decoder_task_process.start()

    blip_vqa_visual_encoder_batch_size,blip_vqa_text_encoder_batch_size,blip_vqa_text_decoder_batch_size=2,8,16
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


    # print(f"len(images): {len(images)}")
    # print(f"len(texts): {len(texts)}")
    # print(len(blip_vqa_visual_encoder_batches))
    # for i in blip_vqa_visual_encoder_batches:
    #     print(len(i))
    # print(len(blip_vqa_text_encoder_batches))
    # for i in blip_vqa_text_encoder_batches:
    #     print(len(i))
    # print(len(blip_vqa_text_decoder_batches))
    # for i in blip_vqa_text_decoder_batches:
    #     print(len(i))
    # exit(0)

    blip_vqa_visual_encoder_batches_queue.put(blip_vqa_visual_encoder_batches,block=False)

    # for _ in range(num_blip_vqa_visual_encoder_batch):
    #     blip_vqa_visual_encoder_batches_return=blip_vqa_visual_encoder_blip_vqa_text_encoder_queue.get()
    #     print(blip_vqa_visual_encoder_batches_return.shape)
    # print("finish...")
    # blip_vqa_visual_encoder_task_process.join()
    # exit(0)
    
    blip_vqa_text_encoder_batches_queue.put(blip_vqa_text_encoder_batches,block=False)
    
    # for _ in range(num_blip_vqa_text_encoder_batch):
    #     blip_vqa_text_encoder_batches_return=blip_vqa_text_encoder_blip_vqa_text_decoder_queue.get()
    #     print(blip_vqa_text_encoder_batches_return.shape)
    # print("finish...")
    # blip_vqa_visual_encoder_task_process.join()
    # blip_vqa_text_encoder_task_process.join()
    # exit(0)

    blip_vqa_text_decoder_batches_queue.put(blip_vqa_text_decoder_batches,block=False)

    batch_count=0
    now_left=None
    for _ in range(num_blip_vqa_text_decoder_batch):
        blip_vqa_text_decoder_batches_return=blip_vqa_text_decoder_batches_return_queue.get()
        print(blip_vqa_text_decoder_batches_return.shape)
    print("finish...")

    blip_vqa_visual_encoder_task_process.join()
    blip_vqa_text_encoder_task_process.join()
    blip_vqa_text_decoder_task_process.join()