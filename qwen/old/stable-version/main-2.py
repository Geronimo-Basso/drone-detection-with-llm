import os
import re
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Load all images
base_directory = '../../../teamspace/studios/this_studio/data/qwen-500'
base_directory_predictions = 'qwen/inference-results/'
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
        x2, y2 = map(int, matches[i+1].strip('()').split(','))

        center_x = (x1 + x2) / 2 / image_width
        center_y = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        normalized_boxes.append(f"{center_x:.10f} {center_y:.10f} {width:.10f} {height:.10f}")
    return normalized_boxes

# Function to draw bounding boxes
def draw_bounding_boxes(image, matches):
    draw = ImageDraw.Draw(image)
    for i in range(0, len(matches), 2):
        x1, y1 = map(int, matches[i].strip('()').split(','))
        x2, y2 = map(int, matches[i+1].strip('()').split(','))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

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

for file_name in images_filenames:
    image_path = file_name
    
    query = tokenizer.from_list_format([
        {'image': file_name},
        {'text': 'Give me the bounding box of drones in the photo, if any exist.'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)

    # Use regex to find all coordinates inside parentheses
    matches = re.findall(r'\(\d+,\d+\)', response)
    
    if matches and len(matches) % 2 == 0:
        # Open the image to get its dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Normalize the coordinates and format as required
        formatted_bounding_boxes = normalize_and_format_bounding_boxes(matches, image_width, image_height)

        print(formatted_bounding_boxes)
        print(matches)

        # Draw bounding boxes
        with Image.open(image_path) as img:
            draw_bounding_boxes(img, matches)
            img_with_boxes_path = os.path.join(base_directory_predictions, Path(file_name).stem + '.jpg')
            img.save(img_with_boxes_path)
        
        image_prediction += 1
    else:
        result = ""
        image_no_prediction += 1

    output_file_path = os.path.join(base_directory_predictions, Path(file_name).stem + '.txt')
    result = '\n'.join(formatted_bounding_boxes) if matches else result

    print(f"Response for {file_name}: {response}. Normalized coordinates result: {result}")
    with open(output_file_path, 'w') as file:
        file.write(result)

print(f"Images with results: {image_prediction}, Images without results: {image_no_prediction}")

# Calculate IoU for predictions
ious = []
for txt_file in txt_filenames:
    ground_truth_boxes = parse_bounding_boxes(txt_file)
    prediction_file = os.path.join(base_directory_predictions, Path(txt_file).stem + '.txt')
    if os.path.exists(prediction_file):
        predicted_boxes = parse_bounding_boxes(prediction_file)
        for gt_box in ground_truth_boxes:
            best_iou = 0
            for pred_box in predicted_boxes:
                gt_coords = [
                    gt_box[0] - gt_box[2] / 2, gt_box[1] - gt_box[3] / 2,
                    gt_box[0] + gt_box[2] / 2, gt_box[1] + gt_box[3] / 2
                ]
                pred_coords = [
                    pred_box[0] - pred_box[2] / 2, pred_box[1] - pred_box[3] / 2,
                    pred_box[0] + pred_box[2] / 2, pred_box[1] + pred_box[3] / 2
                ]
                iou = calculate_iou(gt_coords, pred_coords)
                best_iou = max(best_iou, iou)
            ious.append(best_iou)

if ious:
    average_iou = sum(ious) / len(ious)
    print(f"Average IoU: {average_iou:.4f}")
else:
    print("No valid IoU calculated.")