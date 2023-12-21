import json

json_file = "/datasets/nlvr/test1.json"

examples = [json.loads(line) for line in open(json_file).readlines()]

# for example in examples:
for i in range(20):
    example = examples[i]
    print(
        example["identifier"][:-1] + "img0.png",
        example["identifier"][:-1] + "img1.png",
        example["sentence"],
        example["label"],
    )