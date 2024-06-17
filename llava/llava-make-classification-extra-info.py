import os
import subprocess
import json
import time

input_directory = '/Users/geronimobasso/Desktop/extra/drones/code/computer-vision/originales-400'
output_file = 'output/llava_results_classification_2.json'
dir_list = os.listdir(input_directory)

results = {}
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

def generate_initial_prompt(image_path):
    return f"Does this image contain drones? Answer '1' if it does, or '0' if it does not: {image_path}."

def generate_detailed_prompt(image_path):
    return f"Provide all the information about the drone in the image, {image_path}."

def process_image(image_path):
    initial_prompt = generate_initial_prompt(image_path)
    initial_command = f"ollama run llava prompt '{initial_prompt}'"
    
    try:
        initial_result_model = subprocess.run(initial_command, shell=True, capture_output=True, text=True)
        initial_result = initial_result_model.stdout.strip()

        if "1" in initial_result:
            detailed_prompt = generate_detailed_prompt(image_path)
            detailed_command = f"ollama run llava prompt '{detailed_prompt}'"
            detailed_result_model = subprocess.run(detailed_command, shell=True, capture_output=True, text=True)
            detailed_result = detailed_result_model.stdout.strip()
            return 1, detailed_result
        else:
            return 0, None

    except Exception as e:
        return str(e), None

start_time = time.time()

for filename in dir_list:
    _, file_extension = os.path.splitext(filename)
    if file_extension.lower() == '.jpg':
        image_path = os.path.join(input_directory, filename)
        result, drone_info = process_image(image_path)

        ground_truth_label = get_ground_truth(filename)

        results[filename] = {
            "result": result,
            "ground_truth": ground_truth_label,
            "drone_info": drone_info
        }

        ground_truth_labels.append(ground_truth_label)
        predicted_labels.append(result)

        print(f"{filename}: {result}")

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

# Save results to a JSON file
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)