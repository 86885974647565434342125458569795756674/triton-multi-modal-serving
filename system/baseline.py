import numpy as np
import random
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
            batch_size_list=[random.randint(1,5)*2 for _ in range(request_num)]
            print(f"request_num: {request_num}")
            print(batch_size_list)


            images_batches = [
                np.array([b"/workspace/demos/images/merlion.png",b"/workspace/demos/images/beach.jpg"]*(batch_size//2)) for batch_size in batch_size_list
            ]
            texts_batches = [
                np.array([b"where is it?",b"where is the woman sitting?"]*(batch_size//2)) for batch_size in batch_size_list
            ]
            livings_batches = [
                np.array([random.randint(1, 10),random.randint(1, 10)]*(batch_size//2)) for batch_size in batch_size_list
            ]

            request_ids, batch_nums,images,texts,livings=[],[],[],[],[]
            for i in range(request_num):
                request_id,image_batch,text_batch,living_batch=i,images_batches[i],texts_batches[i],livings_batches[i]
                request_ids.append(request_id)
                images.append(image_batch)
                texts.append(text_batch)
                livings.append(living_batch)
                batch_nums.append(image_batch.shape[0])        

            images=np.concatenate(images, axis=0)
            texts=np.concatenate(texts, axis=0)

            input_queue.put((images,texts))

            results=[]
            request_count=0

            result = output_queue.get()
            print(result)
        print("--------------------------------------------------------------------------")
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("finish...")
