import json
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_iou(pred_box, gt_box):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    pred_box (list): Predicted bounding box [x_min, y_min, width, height]
    gt_box (list): Ground truth bounding box [x_min, y_min, width, height]

    Returns:
    float: IoU value
    """
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
    """
    Calculate IoUs for all images in the dataset.

    Parameters:
    data (list): List of dictionaries with image data

    Returns:
    list: List of tuples (iou, is_true_positive)
    """
    ious = []
    threshold = 0.2  # Define your IoU threshold for true positives
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
    """
    Create a confusion matrix based on IoUs and the threshold.

    Parameters:
    ious (list): List of tuples (iou, is_true_positive)

    Returns:
    None
    """
    y_true = [1] * len(ious)  # All ground truths are positive examples
    y_pred = [int(is_tp) for _, is_tp in ious]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# Load the JSON data
file_path = 'output/results-3.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Calculate IoUs
ious = calculate_ious(data)

# Create Confusion Matrix
create_confusion_matrix(ious)

# If you need to use this script directly, replace 'path_to_your_file.json' with the actual path to your JSON file.
