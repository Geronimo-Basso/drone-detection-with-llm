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


def calculate_iou(pred_box, gt_box):
    x_min_pred, y_min_pred, w_pred, h_pred = pred_box
    x_min_gt, y_min_gt, w_gt, h_gt = gt_box

    x_max_pred = x_min_pred + w_pred
    y_max_pred = y_min_pred + h_pred
    x_max_gt = x_min_gt + w_gt
    y_max_gt = y_min_gt + h_gt

    # Calculate intersection
    x_min_inter = max(x_min_pred, x_min_gt)
    y_min_inter = max(y_min_pred, y_min_gt)
    x_max_inter = min(x_max_pred, x_max_gt)
    y_max_inter = min(y_max_pred, y_max_gt)

    if x_min_inter < x_max_inter and y_min_inter < y_max_inter:
        inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    else:
        inter_area = 0

    # Calculate union
    pred_area = w_pred * h_pred
    gt_area = w_gt * h_gt
    union_area = pred_area + gt_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def calculate_ious(data):
    ious = []
    threshold = 0.5  # Define your IoU threshold for true positives
    for item in data:
        pred_boxes = item['bounding_box_prediction']
        gt_boxes = item['ground_truth_bounding_boxes']

        for gt_box in gt_boxes:
            iou_max = 0
            for pred_box in pred_boxes:
                iou = calculate_iou(pred_box, gt_box)
                iou_max = max(iou_max, iou)
            ious.append((iou_max, iou_max >= threshold))

    return ious


def create_confusion_matrix(ious):
    y_true = [1] * len(ious)  # All ground truths are positive examples
    y_pred = [int(is_tp) for _, is_tp in ious]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    tp = cm[1, 1]
    fn = cm[1, 0]
    total = tp + fn
    percentage = (tp / total) * 100 if total != 0 else 0

    print(f"Drones detected (True Positives): {tp}")
    print(f"No drones detected (False Negatives): {fn}")
    print(f"Percentage of drones correctly detected: {percentage:.2f}%")


def analyse_predictions():
    # Load the JSON data
    file_path = 'output/results-4.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

        # normalize data

    # Calculate IoUs
    ious = calculate_ious(data)

    # Create Confusion Matrix
    create_confusion_matrix(ious)


make_predictions()
#analyse_predictions()
