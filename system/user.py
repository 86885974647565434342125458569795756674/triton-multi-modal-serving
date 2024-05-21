import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import multiprocessing
import time

# the number of events occurring within the specified interval.
# np.random.poisson(lam) lam is avg number
def poisson_request(lambd,total_time):
    request_list=[]
    for _ in range(total_time):
        request_list.append(np.random.poisson(lambd))

    # print(max(request_list),min(request_list),sum(request_list)/len(request_list))
    # plt.bar(range(total_time)[:100], request_list[:100])
    # plt.show()

    return request_list

def azure_request(begin_time,end_time,acc):
    azure_2021 = pd.read_csv("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt", sep=',')
    azure_2021["begin_timestamp"] = azure_2021["end_timestamp"] - azure_2021["duration"]
    
    azure_2021.drop(azure_2021[azure_2021.begin_timestamp < begin_time].index, inplace=True)
    azure_2021.drop(azure_2021[azure_2021.begin_timestamp > end_time].index, inplace=True)
    azure_2021.begin_timestamp = azure_2021.begin_timestamp.astype(int)

    request_list = [0]*((end_time + 1-begin_time)//acc)

    for indx, values in azure_2021['begin_timestamp'].value_counts().items():
        request_list[(indx-begin_time)//acc] += values

    # print(max(request_list),min(request_list),sum(request_list)/len(request_list))
    # plt.bar(range(begin_time,end_time+1)[:100], request_list[:100])
    # plt.show()

    return request_list

def generate_request(workload_name,acc=8,fix=32,lambd=32):
    day_second = 60 * 60 * 24
    begin_time = day_second * 3
    end_time = day_second * 4

    request_list=[]
    if workload_name=="fix":
        request_list=[fix]*((end_time-begin_time+1)//acc)
    elif workload_name=="poisson":
        request_list=poisson_request(lambd,(end_time-begin_time+1)//acc)
    elif workload_name=="azure":
        request_list=azure_request(begin_time,end_time,acc)

    return request_list

def send_and_receive(dataset_dir,json_file,request_queue,processed_results,workload_name):
    try:
        request_list=generate_request(workload_name)
        request_id=0
        request_id_max=len(request_list)-1

        with open(json_file) as f:
            dataset = json.load(f)
        dataset_id=0
        dataset_id_max=len(dataset)-1
        print(request_id_max,dataset_id_max)
        exit(0)

        time_interval=1

        while request_id<=request_id_max and dataset_id<=dataset_id_max:
            start_time=time.time()
            datas_list=dataset[dataset_id:dataset_id+request_list[request_id]]
            dataset_id+=request_list[request_id]
            request_id+=1

            images_batches = [np.array([bytes(os.path.join(dataset_dir, data["image"]), "utf-8") for data in datas]) for datas in datas_list]
        # texts_batches = [np.array([bytes(os.path.join(dataset_dir, data["question"]), "utf-8") for data in datas]) for datas in datas_list]
        # livings_batches = [np.array([1000 for data in datas]) for datas in datas_list]

        #         # with open(write_file,"a") as f:
        #         #     f.write(f"request_num: {request_num}\n")
        #         #     f.write(str(batch_size_list)+"\n")
                
        #         start_time=time.time()
        #         request_queue.put((request_ids,batch_nums,images,texts,livings))

        #         results=[]
        #         end_times=[]
        #         request_count=0
        #         while request_count<request_num:
        #             result = processed_results.get(request_count,None)
        #             if result is not None:
        #                 end_times.append(time.time())
        #                 # results.append(result)
        #                 request_count+=1
                
        #         for request_count in range(request_num):
        #             del processed_results[request_count]

        #         # for e in end_times:
        #         #     print(e-start_time)
        #         if repeat_time==2:
        #             print(f"total time: {end_times[-1]-start_time}")
        #             print(f"avg time: {sum(end_times)/len(end_times)-start_time}")

                # for r in results:
                #     print(r)

                # with open(write_file,"a") as f:
                #     for i in results:
                #         f.write(str(i)+"\n")
        print("--------------------------------------------------------------------------")
    except KeyboardInterrupt:
        pass

if __name__=="__main__":
    dataset_dir = "/workspace/datasets/vqa/"
    json_file = "/workspace/datasets/vqa/vqa_test.json"

    request_queue=multiprocessing.Queue()
    manager = multiprocessing.Manager()
    processed_results = manager.dict()
    
    send_and_receive(dataset_dir,json_file,request_queue,processed_results,"fix")