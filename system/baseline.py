import numpy as np
import random
import time
import multiprocessing
from blip_vqa_process import while_client

if __name__ == "__main__":  
    try:
        # Create a list to hold process objects
        processes = []
        input_queue=multiprocessing.Queue()
        output_queue=multiprocessing.Queue()
        blip_vqa_process = multiprocessing.Process(target=while_client,args=("blip_vqa",input_queue,output_queue))
        processes.append(blip_vqa_process)
        blip_vqa_process.start()

        for _ in range(3):
            request_num=random.randint(1, 10)
            request_num=8
            batch_size_list=[random.randint(1,6) for _ in range(request_num)]
            batch_size_list=[1, 4, 6, 2, 5, 1, 3, 5]
            print(f"request_num: {request_num}")
            print(batch_size_list)


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
            
            images=np.concatenate(images, axis=0)
            texts=np.concatenate(texts, axis=0)
            
            input_queue.put((images,texts))

            result = output_queue.get()
            end_time=time.time()
            print(f"total time: {end_time-start_time}")
            # print(images_batches)
            # print(texts_batches)
            # print(result)
        print("--------------------------------------------------------------------------")
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("finish...")
