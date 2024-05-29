import requests
import json
import time
import re

def explict():
    url = 'http://localhost:8000/v2/repository/index'  # Update with your server's address
    response = requests.post(url)
    model_name=response.json()[0]["name"]
    print(response.json()[0])
    print("---------------------------------------")

    url=f"http://localhost:8000/v2/models/{model_name}/config"
    response=requests.get(url)
    for k,v in response.json().items():
        print(f"{k}:{v}")
    print("---------------------------------------")

    # CUDA_VISIBLE_DEVICES="1" tritonserver --model-repository=/workspace/run_models --model-control-mode=explicit --load-model="blip_vqa"
    url=f"http://localhost:8000/v2/repository/models/{model_name}/load"
    data = {
        "parameters": {
            "config": json.dumps({
                	"name": "blip_vqa",
                    "backend": "python",
                    "max_batch_size": "15",
                    "input": [
                        {
                            "name": "INPUT0",
                            "data_type": "TYPE_STRING",
                            "dims": [ -1 ]
                        },
                        {
                            "name": "INPUT1",
                            "data_type": "TYPE_STRING",
                            "dims": [ -1 ]
                        }
                    ],
                    "output": [
                        {
                            "name": "OUTPUT0",
                            "data_type": "TYPE_STRING",
                            "dims": [ -1 ]
                        }
                    ],
                    "instance_group":[{ "kind": "KIND_GPU", "gpus":[0] }]
            })
        }
    }
    print(f"begin post:{time.time()}")
    response=requests.post(url,json=data)
    print(f"after post:{time.time()}")
    # begin post:1716970099.456369
    # after post:1716970117.2166884
    print(response.status_code)
    print("---------------------------------------")

    url=f"http://localhost:8000/v2/models/{model_name}/config"
    response=requests.get(url)
    for k,v in response.json().items():
        print(f"{k}:{v}")
    print("---------------------------------------")

    url = 'http://localhost:8000/v2/repository/index'  # Update with your server's address
    response = requests.post(url)
    print(response.json()[0])
    print("---------------------------------------")

    print("finish")

def update_max_batch_size(file_path, new_max_batch_size):
    # Read the existing config.pbtxt file
    with open(file_path, 'r') as file:
        config_content = file.read()

# 'name: "blip_vqa"\nbackend: "python"\n\nmax_batch_size: 0\ninput [\n  {\n    name: "INPUT0"\n    data_type: TYPE_STRING\n    dims: [ -1 ]\n  }\n]\ninput [\n  {\n    name: "INPUT1"\n    data_type: TYPE_STRING\n    dims: [ -1 ]\n  }\n]\noutput [\n  {\n    name: "OUTPUT0"\n    data_type: TYPE_STRING\n    dims: [ -1 ]\n  }\n]\n\ninstance_group [{ kind: KIND_GPU, gpus:[0] }]\n'
    
    # Regular expression to find and replace max_batch_size
    max_batch_size_pattern = re.compile(r'max_batch_size\s*:\s*\d+')
    new_max_batch_size_str = f'max_batch_size: {new_max_batch_size}'
    updated_config_content = re.sub(max_batch_size_pattern, new_max_batch_size_str, config_content)

    # Write the updated config back to the file
    with open(file_path, 'w') as file:
        file.write(updated_config_content)


def poll():
    url = 'http://localhost:8000/v2/repository/index'  # Update with your server's address
    response = requests.post(url)
    model_name=response.json()[0]["name"]
    print(response.json()[0])
    print("---------------------------------------")

    url=f"http://localhost:8000/v2/models/{model_name}/config"
    response=requests.get(url)
    for k,v in response.json().items():
        print(f"{k}:{v}")
    print("---------------------------------------")

    # CUDA_VISIBLE_DEVICES="1" tritonserver --model-repository=/workspace/run_models --model-control-mode=poll --repository-poll-secs=1
    url=f"http://localhost:8000/v2/repository/models/{model_name}/load"
    file_path = f'/workspace/run_models/{model_name}/config.pbtxt'
    new_max_batch_size = 32
    print(f"begin post:{time.time()}")
    update_max_batch_size(file_path, new_max_batch_size)
    # begin post:1716971600.5196402
    #       time:1716971618.9656885
    print("---------------------------------------")

    # url=f"http://localhost:8000/v2/models/{model_name}/config"
    # response=requests.get(url)
    # for k,v in response.json().items():
    #     print(f"{k}:{v}")
    # print("---------------------------------------")

    # url = 'http://localhost:8000/v2/repository/index'  # Update with your server's address
    # response = requests.post(url)
    # print(response.json()[0])
    # print("---------------------------------------")

    print("finish")

    # reading config.pbtxt will also reload
    # curl -X POST localhost:8000/v2/repository/index
    # curl localhost:8000/v2/models/blip_vqa/config
    # new_max_batch_size = 0
    # update_max_batch_size(file_path, new_max_batch_size)

if __name__=="__main__":
    explict()
    # poll() no