"""
CM:
TP: Exists drone in GT and detected drone in P.
FN: Exists drone in GT and no detected drone in P.
FP: No drone exists in GT and detected drone in P.(0)
TN: No drone exists in GT and no detected drone in P. (0)
"""
import os
import subprocess
import json
import time

input_directory = '/Users/geronimobasso/Desktop/extra/drones/code/computer-vision/originales-400'
dir_list = os.listdir(input_directory)

results = {}
yes_count = 0
no_count = 0
ground_truth_labels = []
predicted_labels = []


def get_ground_truth(filename):
    base_filename, _ = os.path.splitext(filename)
    txt_file = os.path.join(input_directory, base_filename + '.txt')
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            content = file.read().strip()
            if len(content) > 0:
                return 1
    return 0


def generate_prompt(image_path):
    prompt = f"Does this image contain drones? Answer '1' if it does, or '0' if it does not: {image_path}."
    return prompt


def process_image(image_path):
    prompt = generate_prompt(image_path)
    command = f"ollama run llava prompt '{prompt}'"
    try:
        result_model = subprocess.run(command, shell=True, capture_output=True, text=True)
        result = result_model.stdout.strip()

        if "1" in result.lower():
            return 1
        else:
            return 0

    except Exception as e:
        return str(e)

start_time = time.time()

for filename in dir_list:
    _, file_extension = os.path.splitext(filename)
    if file_extension.lower() == '.jpg':
        image_path = os.path.join(input_directory, filename)
        result = process_image(image_path)

        ground_truth_label = get_ground_truth(filename)

        results[filename] = {"result": result, "ground_truth": ground_truth_label}

        ground_truth_labels.append(ground_truth_label)
        predicted_labels.append(result)

        print(f"{filename}: {result}")

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

# Save results to a JSON file
output_file = 'output/llava_results_classification.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
