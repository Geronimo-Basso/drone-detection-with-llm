"""
Simplified Detection Approach:
	1.	Objective:
	•	Simply checks if there are any bounding boxes present in both ground truth and prediction.
	2.	Logic:
	•	True Positives (TP): If both the ground truth and prediction have bounding boxes, it is considered a true positive (drone detected in both).
	•	False Positives (FP): If the prediction has bounding boxes but the ground truth does not, it is considered a false positive (drone detected in prediction but not in ground truth).
	•	False Negatives (FN): If the ground truth has bounding boxes but the prediction does not, it is considered a false negative (drone present in ground truth but not detected in prediction).
"""
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_bounding_box_coordinates(bbox_info, width, height):
    if len(bbox_info) != 5:
        return None
    _, center_x, center_y, bbox_width, bbox_height = map(float, bbox_info)
    center_x *= width
    center_y *= height
    bbox_width *= width
    bbox_height *= height
    top_left_x = int(center_x - (bbox_width / 2))
    top_left_y = int(center_y - (bbox_height / 2))
    bottom_right_x = int(center_x + (bbox_width / 2))
    bottom_right_y = int(center_y + (bbox_height / 2))
    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Directory containing the images and bounding box files
input_directory = '../../../teamspace/studios/this_studio/data/qwen-500'
prediction_directory = 'qwen/inference-results/'

iou_scores = []
tp = 0
fp = 0
fn = 0
processed_images = 0

for filename in os.listdir(input_directory):
    if filename.endswith('.jpe') or filename.endswith('.jpg'):
        image_path = os.path.join(input_directory, filename)
        ground_truth_path = os.path.join(input_directory, filename.replace('.jpe', '.txt').replace('.jpg', '.txt'))
        prediction_path = os.path.join(prediction_directory, filename.replace('.jpe', '.txt').replace('.jpg', '.txt'))

        if os.path.exists(ground_truth_path) and os.path.exists(prediction_path):
            with Image.open(image_path) as img:
                width, height = img.size

            with open(ground_truth_path, 'r') as f:
                gt_bboxes = [line.strip().split() for line in f.readlines()]
            gt_boxes = [get_bounding_box_coordinates(bbox_info, width, height) for bbox_info in gt_bboxes if get_bounding_box_coordinates(bbox_info, width, height) is not None]

            with open(prediction_path, 'r') as f:
                pred_bboxes = [line.strip().split() for line in f.readlines()]
            pred_boxes = [get_bounding_box_coordinates(bbox_info, width, height) for bbox_info in pred_bboxes if get_bounding_box_coordinates(bbox_info, width, height) is not None]

            gt_has_drone = len(gt_boxes) > 0
            pred_has_drone = len(pred_boxes) > 0

            if gt_has_drone and pred_has_drone:
                # If both have drones, consider it a true positive
                tp += 1
            elif pred_has_drone and not gt_has_drone:
                # If the prediction has drones but ground truth does not, false positive
                fp += 1
            elif gt_has_drone and not pred_has_drone:
                # If the ground truth has drones but prediction does not, false negative
                fn += 1

            processed_images += 1
            print(f"Finished processing {filename}")

# Print TP, FP, FN counts
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

# Confusion Matrix for Images
y_true = ['TP'] * tp + ['FN'] * fn + ['TN'] * (processed_images - tp - fp - fn)
y_pred = ['TP'] * tp + ['FP'] * fp + ['FN'] * fn + ['TN'] * (processed_images - tp - fp - fn)
conf_matrix = confusion_matrix(y_true, y_pred, labels=['TP', 'FP', 'FN', 'TN'])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred TP', 'Pred FP', 'Pred FN', 'Pred TN'],
            yticklabels=['Actual TP', 'Actual FP', 'Actual FN', 'Actual TN'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Images')
plt.show()

print(f"Finished processing all {processed_images} images")