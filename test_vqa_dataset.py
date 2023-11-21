import json

json_file = "datasets/vqa/test.json"
with open(json_file) as f:
    dataset = json.load(f)

for item in dataset:
    print(item["image"], item["question"])

print(len(dataset))
