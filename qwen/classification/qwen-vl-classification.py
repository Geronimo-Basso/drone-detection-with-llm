import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

# Load all images
base_directory = '/teamspace/studios/this_studio/drone-detection-with-llm/originales-400'
base_directory_predictions = '/teamspace/studios/this_studio/drone-detection-with-llm/qwen/classification/output/'
image_count = 0
images_filenames = []

with os.scandir(base_directory) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.lower().endswith('.jpg'):
            image_count += 1
            images_filenames.append(entry.path)

print(f"Images in total: {len(images_filenames)}")

start_time = time.time()
print(f"Start timer")

# Download the model
torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Processing images
image_with_drone = 0
image_without_drone = 0
results = []

for file_name in images_filenames:
    image_path = file_name

    query = tokenizer.from_list_format([
        {'image': file_name},
        {"text": "Is there a drone in this photo? Answer only yes o no"}
    ])
    response, history = model.chat(tokenizer, query=query, history=None)

    print(response)

    if "yes" in response.lower():
        image_with_drone += 1
        drone_present = True
    else:
        image_without_drone += 1
        drone_present = False

    result = {
        'image': Path(file_name).name,
        'drone_present': drone_present
    }
    results.append(result)

    print(f"Response for {file_name}: {response}. Drone present: {drone_present}")

# Save results to JSON
output_json_path = os.path.join(base_directory_predictions, 'classification_results.json')
with open(output_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

finish_time = time.time()
elapsed_time = finish_time - start_time
print(f"Images with drone: {image_with_drone}, Images without drone: {image_without_drone}")
print(f"Total time: {elapsed_time}")
