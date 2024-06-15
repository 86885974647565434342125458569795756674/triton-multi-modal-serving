import requests
import json
import time
import sys

bs=4
model_name="blip_vqa_visual_encoder"

def change_batch_size(bs,model_name):
    url=f"http://localhost:8000/v2/batch/{model_name}"
    data = {
        "max_batch_size": bs
    }
#    print(f"begin post:{time.time()}")
    response=requests.post(url,json=data)
#    print(f"after post:{time.time()}")
    # begin post:1716970099.456369
    # after post:1716970117.2166884
    print(response.status_code)

if __name__=="__main__":
    
    if len(sys.argv) > 1:
        bs = int(sys.argv[1])
        if len(sys.argv)>2:
            model_name=sys.argv[2]

    print(f"set batch size:{bs}")
    change_batch_size(bs,model_name)
