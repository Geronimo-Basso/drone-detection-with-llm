from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Convertir coordenadas de caja delimitadora normalizadas a valores en píxeles
def get_bounding_box_coordinates(bbox_info, width, height, is_ground_truth=True):
    if is_ground_truth and len(bbox_info) != 5:
        return None
    if not is_ground_truth and len(bbox_info) != 4:
        return None
    
    if is_ground_truth:
        _, center_x, center_y, bbox_width, bbox_height = map(float, bbox_info)
    else:
        center_x, center_y, bbox_width, bbox_height = map(float, bbox_info)

    center_x *= width
    center_y *= height
    bbox_width *= width
    bbox_height *= height
    top_left_x = int(center_x - (bbox_width / 2))
    top_left_y = int(center_y - (bbox_height / 2))
    bottom_right_x = int(center_x + (bbox_width / 2))
    bottom_right_y = int(center_y + (bbox_height / 2))
    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

# Calcular IoU entre dos cajas delimitadoras
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

# Directorios que contienen las imágenes y archivos de cajas delimitadoras
input_directory = '../../../teamspace/studios/this_studio/data/qwen-500'
prediction_directory = 'qwen/inference-results-6/'

iou_threshold = 0.3

iou_scores = []
tp = 0
fp = 0
fn = 0
tn = 0
processed_images = 0

# Iterar a través de cada imagen en el directorio de entrada
for filename in os.listdir(input_directory):
    if filename.endswith('.jpe') or filename.endswith('.jpg'):
        image_path = os.path.join(input_directory, filename)
        ground_truth_path = os.path.join(input_directory, filename.replace('.jpe', '.txt').replace('.jpg', '.txt'))
        prediction_path = os.path.join(prediction_directory, filename.replace('.jpe', '.txt').replace('.jpg', '.txt'))

        if os.path.exists(ground_truth_path) and os.path.exists(prediction_path):
            with Image.open(image_path) as img:
                width, height = img.size

            # Leer las cajas delimitadoras de verdad de terreno y predicción
            with open(ground_truth_path, 'r') as f:
                gt_bboxes = [line.strip().split() for line in f.readlines()]
            gt_boxes = [get_bounding_box_coordinates(bbox_info, width, height, is_ground_truth=True) for bbox_info in gt_bboxes if get_bounding_box_coordinates(bbox_info, width, height, is_ground_truth=True) is not None]

            with open(prediction_path, 'r') as f:
                pred_bboxes = [line.strip().split() for line in f.readlines()]
            pred_boxes = [get_bounding_box_coordinates(bbox_info, width, height, is_ground_truth=False) for bbox_info in pred_bboxes if get_bounding_box_coordinates(bbox_info, width, height, is_ground_truth=False) is not None]

            gt_has_drone = len(gt_boxes) > 0
            pred_has_drone = False

            for pred_box in pred_boxes:
                for gt_box in gt_boxes:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > iou_threshold:
                        pred_has_drone = True
                        break
                if pred_has_drone:
                    break

            if gt_has_drone and pred_has_drone:
                tp += 1
            elif gt_has_drone and not pred_has_drone:
                fn += 1
            elif not gt_has_drone and pred_has_drone:
                fp += 1
            elif not gt_has_drone and not pred_has_drone:
                tn += 1

            processed_images += 1
            print(f"Finished processing {filename}")

# Imprimir recuentos de TP, FP, FN, TN
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")

# Matriz de confusión para imágenes
y_true = ['TP'] * tp + ['FN'] * fn + ['TN'] * tn + ['FP'] * fp
y_pred = ['TP'] * tp + ['FN'] * fn + ['TN'] * tn + ['FP'] * fp
conf_matrix = confusion_matrix(y_true, y_pred, labels=['TP', 'FP', 'FN', 'TN'])

# Graficar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred TP', 'Pred FP', 'Pred FN', 'Pred TN'],
            yticklabels=['Actual TP', 'Actual FP', 'Actual FN', 'Actual TN'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Images')
plt.show()

print(f"Finished processing all {processed_images} images")