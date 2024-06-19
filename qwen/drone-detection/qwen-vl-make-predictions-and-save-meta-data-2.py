import os
import re
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import time
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Function to parse bounding boxes from text files
def parse_bounding_boxes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    bounding_boxes = []
    for line in lines:
        if line.strip() != "0 0 0 0":
            coords = list(map(float, line.strip().split()))  # Ignore the leading '0'
            bounding_boxes.append(coords)
    return bounding_boxes

def make_predictions():
    # Load all images
    base_directory = '/teamspace/studios/this_studio/drone-detection-with-llm/datasets/originales-400-txt-edit'
    base_directory_predictions = '/teamspace/studios/this_studio/drone-detection-with-llm/qwen/drone-detection/output'

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

    start_time = time.time()
    print(f"Start timer")

    # Download the model
    torch.manual_seed(1234)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # Processing images
    image_prediction = 0
    image_no_prediction = 0
    results = []

    for file_name in images_filenames:
        image_path = file_name
        ground_truth_path = file_name.replace('.jpg', '.txt')

        query = tokenizer.from_list_format([
            {'image': file_name},
            {"text": "Give me the bounding box of drones in the photo, if any exist."}
        ])
        response, history = model.chat(tokenizer, query=query, history=None)

        # Use regex to find all coordinates inside parentheses
        matches = re.findall(r'\(\d+,\d+\)', response)

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        if matches and len(matches) % 2 == 0:
            image_prediction += 1
        else:
            image_no_prediction += 1

        # Parse ground truth bounding boxes
        if os.path.exists(ground_truth_path):
            ground_truth_bounding_boxes = parse_bounding_boxes(ground_truth_path)
        else:
            ground_truth_bounding_boxes = []

        result = {
            'image': Path(file_name).name,
            'width': image_width,
            'height': image_height,
            'bounding_box_prediction': matches,
            'ground_truth_bounding_boxes': ground_truth_bounding_boxes
        }
        results.append(result)

        print(f"Response for {file_name}: {matches}.")

    # Save results to JSON
    output_json_path = os.path.join(base_directory_predictions, 'results-5.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    finish_time = time.time()
    elapsed_time = finish_time - start_time
    print(f"Images with results: {image_prediction}, Images without results: {image_no_prediction}")
    print(f"Total time: {elapsed_time}")

# Function to normalize bounding boxes
def normalize_bounding_box(box, image_width, image_height):
    x1, y1 = box[0]
    x2, y2 = box[1]
    center_x = (x1 + x2) / 2 / image_width
    center_y = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return [center_x, center_y, width, height]

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

# Function to evaluate predictions
def evaluate_predictions(predictions, iou_threshold=0.5):
    total_true = 0
    total_false = 0

    for entry in predictions:
        image = entry['image']
        image_width = entry['width']
        image_height = entry['height']
        predicted_boxes = entry['bounding_box_prediction']
        ground_truth_boxes = entry['ground_truth_bounding_boxes']

        # Skip if there are no predicted bounding boxes
        if not predicted_boxes:
            entry['correct_detection'] = False
            total_false += 1
            continue

        # Convert predicted bounding boxes to normalized format
        normalized_predicted_boxes = [
            normalize_bounding_box(
                [(int(coord.split(',')[0][1:]), int(coord.split(',')[1][:-1])) for coord in predicted_boxes],
                image_width, image_height
            )
        ]

        # Calculate IoU for each predicted box with each ground truth box
        detection_correct = False
        for pred_box in normalized_predicted_boxes:
            for gt_box in ground_truth_boxes:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    entry['correct_detection'] = True
                    detection_correct = True
                    total_true += 1
                    break
            if detection_correct:
                break
        else:
            entry['correct_detection'] = False
            total_false += 1

    print(f"Total True: {total_true}")
    print(f"Total False: {total_false}")

    return predictions

def analyse_predictions():
    # Load predictions from JSON
    predictions_path = 'output/results-5.json'
    with open(predictions_path, 'r') as file:
        predictions = json.load(file)

    # Evaluate predictions
    evaluated_predictions = evaluate_predictions(predictions)

    # Save evaluated predictions to a new JSON file
    output_path = 'output/results-5-analysis.json'
    with open(output_path, 'w') as file:
        json.dump(evaluated_predictions, file, indent=4)

    print(f"Evaluated predictions saved to {output_path}")


#make_predictions()
analyse_predictions()
