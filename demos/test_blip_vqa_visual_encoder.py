import numpy as np
import time
import tritonclient.http as httpclient
from tritonclient.utils import *
import numpy as np
import asyncio

root_path='/dynamic_batch/triton-multi-modal-serving'
model_name = "blip_vqa_visual_encoder"
user_num=30
input0_data=np.array([[root_path.encode('utf-8')+b"/demos/images/merlion.png"]])

clients=[]
inputss=[]
outputss=[]
responses=[]
times=[]
for _ in range(user_num):
    clients.append(httpclient.InferenceServerClient("localhost:8000"))

    inputss.append([
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
    ])

    inputss[-1][0].set_data_from_numpy(input0_data)

    outputss.append([
        httpclient.InferRequestedOutput("OUTPUT0"),
        ])

async def infer_async(client, model_name, inputs, outputs):
#    print(time.time())
    response=await asyncio.to_thread(client.infer, model_name, inputs, outputs=outputs)
    return response,time.time()

async def main_async():
    tasks = [infer_async(clients[i], model_name, inputss[i], outputss[i]) for i in range(user_num)]
    responses_times = await asyncio.gather(*tasks)
    for rt in responses_times:
        responses.append(rt[0])
        times.append(rt[1])

def main():
    for i in range(user_num):
        responses.append(clients[i].infer(model_name, inputss[i], outputs=outputss[i]))
        times.append(time.time())

if __name__=="__main__":
    #main()
    asyncio.run(main_async())

    output0_datas = np.concatenate([response.as_numpy("OUTPUT0") for response in responses],axis=0)
    print("OUTPUT0 ({})".format(output0_datas.shape))
    times.sort()
    for t in times:
        print(t)
    
    for client in clients:
        client.close()
