import http.server
import queue
import time
import numpy as np
import json
import random
import multiprocessing
import os
from blip_vqa_process import change_batch_size,blip_vqa_process

def send_and_receive(request_queue,request_events,processed_results):
    try:
        write_file="/workspace/send_and_receive.txt"
        if(os.path.isfile(write_file)):    
            os.remove(write_file)
        for _ in range(3):
            request_num=random.randint(1, 10)
            request_num=8
            batch_size_list=[random.randint(1,6) for _ in range(request_num)]
            batch_size_list=[1, 4, 6, 2, 5, 1, 3, 5]
            
            with open(write_file,"a") as f:
                f.write(f"request_num: {request_num}\n")
                f.write(str(batch_size_list)+"\n")

            images_batches = [
                np.array([b"/workspace/demos/images/acorns_1.jpg",b"/workspace/demos/images/acorns_6.jpg",b"/workspace/demos/images/beach.jpg",b"/workspace/demos/images/ex0_0.jpg",b"/workspace/demos/images/ex0_1.jpg",b"/workspace/demos/images/merlion.png"][:batch_size]) for batch_size in batch_size_list
            ]
            texts_batches = [
                np.array([b"describe the picture detailly",b"tell me the detail of the photo",b"describe the dog detailly and beautifully",b"tell me the relationship of the dog and human",b"tell me everything about this dog",b"please describe the landscape detailly and gracefully"][:batch_size]) for batch_size in batch_size_list
            ]
            livings_batches = [
                np.array([1000]*batch_size) for batch_size in batch_size_list
            ]
            
            request_ids, batch_nums,images,texts,livings=[],[],[],[],[]
            for i in range(request_num):
                request_id,image_batch,text_batch,living_batch=i,images_batches[i],texts_batches[i],livings_batches[i]
                request_ids.append(request_id)
                images.append(image_batch)
                texts.append(text_batch)
                livings.append(living_batch)
                batch_nums.append(image_batch.shape[0])        
            
            start_time=time.time()
            request_queue.put((request_ids,batch_nums,images,texts,livings))

            results=[]
            end_times=[]
            request_count=0
            while request_count<request_num:
                result = processed_results.get(request_count,None)
                if result is not None:
                    end_times.append(time.time())
                    del processed_results[request_count]
                    results.append(result)
                    request_count+=1

            for e in end_times:
                print(e-start_time)
            print(f"avg time: {sum(end_times)/len(end_times)-start_time}")
            
            print()
            # for r in results:
            #     print(r)

            with open(write_file,"a") as f:
                for i in results:
                    f.write(str(i)+"\n")
        print("--------------------------------------------------------------------------")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try:
        # Create a list to hold process objects
        processes = []

        batch_size_queue=multiprocessing.Queue()
        # queue is not the fastest way, maybe pipe
        time_interval=1

        change_batch_size_process = multiprocessing.Process(target=change_batch_size, args=(batch_size_queue,time_interval))
        processes.append(change_batch_size_process)
        change_batch_size_process.start()

        request_queue=multiprocessing.Queue()
        manager = multiprocessing.Manager()
        request_events=manager.dict()
        processed_results = manager.dict()
        
        blip_vqa_process_process = multiprocessing.Process(target=blip_vqa_process, args=(request_queue,request_events,processed_results,batch_size_queue,))
        processes.append(blip_vqa_process_process)
        blip_vqa_process_process.start()
        
        send_and_receive_process = multiprocessing.Process(target=send_and_receive, args=(request_queue,request_events,processed_results,))
        processes.append(send_and_receive_process)
        send_and_receive_process.start()
        
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("finish...")
