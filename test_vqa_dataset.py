import json

dataset_url = "/workspace/datasets/vqa/anotations.json"
with open(dataset_url) as f:
    dataset = json.load(f)
for item in dataset:
    print(item["image"], item["question"])

print(len(dataset))
