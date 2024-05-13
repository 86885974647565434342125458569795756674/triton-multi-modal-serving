import http.server
import queue
import time
import numpy as np
import json
import random
import multiprocessing
import os
from blip_vqa_process import change_batch_size,blip_vqa_process_queue

def send_and_receive(request_queue,request_events,processed_results):
    write_file="/workspace/send_and_receive.txt"
    try:
        while True:
            request_num=random.randint(1, 10)
            batch_size_list=[random.randint(1,10) for _ in range(request_num)]

            if(os.path.isfile(write_file)):    
                os.remove(write_file)
            with open(write_file,"a") as f:
                f.write(f"request_num: {request_num}\n")
                f.write(str(batch_size_list)+"\n")

            images = [
                np.array([b"/workspace/demos/images/merlion.png"]*batch_size) for batch_size in batch_size_list
            ]
            texts = [
                np.array([b"where is the woman sitting?"]*batch_size) for batch_size in batch_size_list
            ]

            for i in range(request_num):
                # Put the request in the queue
                request_queue.put((i,images[i],texts[i]))

            results=[]
            request_count=0
            while request_count<request_num:
                result = processed_results.get(request_count,None)
                if result is not None:
                    del processed_results[request_count]
                    results.append(result)
                    request_count+=1
                    
            with open(write_file,"a") as f:
                for i in results:
                    f.write(str(i)+"\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        batch_size_queue=multiprocessing.Queue()
        # queue is not the fastest way, maybe pipe
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
        
        send_and_receive_process = multiprocessing.Process(target=send_and_receive, args=(request_queue,request_events,processed_results,))
        processes.append(send_and_receive_process)
        send_and_receive_process.start()
        
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()

