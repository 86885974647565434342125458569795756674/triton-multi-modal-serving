import requests
import json
import time
import re


def change_batch_size(bs,model_name):
    url=f"http://localhost:8000/v2/batch/{model_name}"
    data = {
        "max_batch_size": bs
    }
    print(f"begin post:{time.time()}")
    response=requests.post(url,json=data)
    print(f"after post:{time.time()}")
    # begin post:1716970099.456369
    # after post:1716970117.2166884
    print(response.status_code)
    print("finish")

if __name__=="__main__":
    model_name="blip_vqa_visual_encoder"
    for bs in [2,4,8,16]:
        change_batch_size(bs,model_name)
        break
