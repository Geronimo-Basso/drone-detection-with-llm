import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Load all images
base_directory = '../../../teamspace/studios/this_studio/data/qwen-500'
base_directory_predictions = 'qwen/inference-results-7/'
image_count = 0
images_filenames = []
txt_count = 0
txt_filenames = []

with os.scandir(base_directory) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.lower().endswith('.jpg'):
            image_count += 1
            images_filenames.append(entry.path)
        elif entry.is_file() and entry.name.lower().endswith('.txt'):
            txt_count += 1
            txt_filenames.append(entry.path)

print(f"Images in total: {len(images_filenames)}")
print(f"Text files in total: {len(txt_filenames)}")

# Download the model
torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# Processing images
image_prediction = 0
image_no_prediction = 0

for file_name in images_filenames:
    image_path = file_name

    query = tokenizer.from_list_format([
        {'image': file_name},
        {
            "text": "Detect all drones in the image and provide their bounding box coordinates in the format [x_center, y_center, width, height] as normalized values between 0 and 1."}
        # {'text': 'Give me the bounding box of drones in the photo, if any exist.'}, #Inference results
        # {'text': '如果有的话，请给我照片中无人机的边界框'}, #Inference results 2
        # {'text': '检测无人机'}, #Inference results 3
        # {'text': 'Detect drones in the image.' # Images with results: 45, Images without results: 455
        # {'text': 'Detect the drones in the photo and give me the bounding box'}, # Inference results 5
        # { "text": "Please detect all the drones in the provided photo and return the bounding box coordinates for each detected drone. The coordinates should be in the format [x_center, y_center, width, height], normalized between 0 and 1. Here's an example of the expected output format for a single drone detection:\n\n{\n  \"drones\": [\n    {\"x_center\": 0.5, \"y_center\": 0.5, \"width\": 0.2, \"height\": 0.1}\n  ]\n}\n\nIf there are multiple drones, include all of them in the list."}  # Inference results 6
    ])
    response, history = model.chat(tokenizer, query=query, history=None)

    print(response)
