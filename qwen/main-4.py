import os
import re
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import json

# Load all images
base_directory = '../originales-500'
base_directory_predictions = '/output'
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


# Function to normalize and format bounding boxes
def normalize_and_format_bounding_boxes(matches, image_width, image_height):
    normalized_boxes = []
    for i in range(0, len(matches), 2):
        x1, y1 = map(int, matches[i].strip('()').split(','))
        x2, y2 = map(int, matches[i + 1].strip('()').split(','))

        center_x = (x1 + x2) / 2 / image_width
        center_y = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        normalized_boxes.append(f"{center_x:.10f} {center_y:.10f} {width:.10f} {height:.10f}")
    return normalized_boxes

# Function to parse bounding boxes from text files
def parse_bounding_boxes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    bounding_boxes = []
    for line in lines:
        if line.strip() != "0 0 0 0":
            coords = list(map(float, line.strip().split()[1:]))  # Ignore the leading '0'
            bounding_boxes.append(coords)
    return bounding_boxes

# Processing images
image_prediction = 0
image_no_prediction = 0
results = []

for file_name in images_filenames:
    image_path = file_name

    query = tokenizer.from_list_format([
        {'image': file_name},
        {
            "text": "Detect all drones in the image and provide their bounding box coordinates in the format [x_center, y_center, width, height] as normalized values between 0 and 1."}
    ])
    response, history = model.chat(tokenizer, query=query, history=None)

    print(response)

    # Use regex to find all coordinates inside parentheses
    matches = re.findall(r'\(\d+,\d+\)', response)

    if matches and len(matches) % 2 == 0:
        # Open the image to get its dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Normalize the coordinates and format as required
        formatted_bounding_boxes = normalize_and_format_bounding_boxes(matches, image_width, image_height)

        image_prediction += 1
    else:
        formatted_bounding_boxes = []
        image_no_prediction += 1

    result = {'image': Path(file_name).name, 'bounding_box_prediction': formatted_bounding_boxes}
    results.append(result)

    print(f"Response for {file_name}: {response}. Normalized coordinates result: {formatted_bounding_boxes}")

# Save results to JSON
output_json_path = os.path.join(base_directory_predictions, 'results.json')
with open(output_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Images with results: {image_prediction}, Images without results: {image_no_prediction}")