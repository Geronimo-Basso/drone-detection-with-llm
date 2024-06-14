import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, fbeta_score, log_loss, precision_recall_curve, \
    average_precision_score

# Load the results from JSON file
input_file = 'output/detection_results_2.json'
with open(input_file, 'r') as f:
    results = json.load(f)

# Prepare lists for ground truth and predicted labels
ground_truth_labels = []
predicted_labels = []

for filename, data in results.items():
    ground_truth_labels.append(data['ground_truth'])
    predicted_labels.append(data['result'])

# Calculate confusion matrix and other metrics
cm = confusion_matrix(ground_truth_labels, predicted_labels)
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
precision = precision_score(ground_truth_labels, predicted_labels, zero_division=0)
recall = recall_score(ground_truth_labels, predicted_labels, zero_division=0)
f1 = f1_score(ground_truth_labels, predicted_labels, zero_division=0)

# Additional metrics
mcc = matthews_corrcoef(ground_truth_labels, predicted_labels)
kappa = cohen_kappa_score(ground_truth_labels, predicted_labels)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
balanced_acc = balanced_accuracy_score(ground_truth_labels, predicted_labels)
f2_score = fbeta_score(ground_truth_labels, predicted_labels, beta=2, zero_division=0)

# Check if there are at least two classes in ground_truth_labels
if len(set(ground_truth_labels)) > 1:
    logloss = log_loss(ground_truth_labels, [0.9 if pred == 1 else 0.1 for pred in predicted_labels])
    auc = roc_auc_score(ground_truth_labels, predicted_labels)
    average_precision = average_precision_score(ground_truth_labels, predicted_labels)
else:
    logloss = None
    auc = None
    average_precision = None

precision_curve, recall_curve, _ = precision_recall_curve(ground_truth_labels, predicted_labels)

# Print metrics
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC: {auc:.2f}" if auc is not None else "AUC: Not applicable (only one class present)")
print(f"Matthews Correlation Coefficient: {mcc:.2f}")
print(f"Cohen’s Kappa: {kappa:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Balanced Accuracy: {balanced_acc:.2f}")
print(f"F2 Score: {f2_score:.2f}")
print(f"Log Loss: {logloss:.2f}" if logloss is not None else "Log Loss: Not applicable (only one class present)")
print(
    f"Average Precision Score: {average_precision:.2f}" if average_precision is not None else "Average Precision Score: Not applicable (only one class present)")
print(f"Total processed images: {len(ground_truth_labels)}")
