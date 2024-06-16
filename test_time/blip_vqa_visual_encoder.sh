#!/bin/bash

root_path=/dynamic_batch/triton-multi-modal-serving
file_name=blip_vqa_visual_encoder
if [ -f "${root_path}/${file_name}_time.txt" ]; then
	rm -rf "${root_path}/${file_name}_time.txt"
fi
# Array of values for i
values=(2 4 8 16 32)

# Loop through each value and execute the Python script
for i in "${values[@]}"
do
	python ${root_path}/models/test/test_${file_name}.py "$i"
done

python ${root_path}/test_time/${file_name}.py

mv ${root_path}/${file_name}_time.png ../../
