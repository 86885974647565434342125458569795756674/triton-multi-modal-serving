import json
import sys
json_file = "/workspace/datasets/nlvr2/test1.json"

import ruamel.yaml as yaml
sys.path.append("BLIP")
from data import create_dataset, create_sampler, create_loader

config_root="BLIP/configs/nlvr.yaml"
yaml = yaml.YAML(typ='rt')
config = yaml.load(open(config_root, 'r'))
ann_root='/trash'
config['ann_root']=ann_root
dataset = create_dataset('nlvr', config)[-1]
dataset.annotation = [json.loads(line) for line in open(json_file).readlines()]
print(dataset.annotation[32])
examples=dataset.annotation
# for example in examples:
for i in range(8):
    example = examples[i]
    print(
        example["identifier"][:-1] + "img0.png",
        example["identifier"][:-1] + "img1.png",
        example["sentence"],
        example["label"],
    )
