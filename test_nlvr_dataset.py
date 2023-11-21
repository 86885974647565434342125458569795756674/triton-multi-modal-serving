import json

json_file = "datasets/nlvr/test1.json"

examples = [json.loads(line) for line in open(json_file).readlines()]

with open("/tmp/tmp.txt", "w") as f:
    for example in examples:
        print(
            example["identifier"] + "img0.png",
            example["identifier"] + "img1.png",
            example["sentence"],
            file=f,
        )
    print(len(examples), file=f)
