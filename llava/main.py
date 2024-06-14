"""
Code working correctly, just need to address the fast that the model does not return 0 or 1 always. Also, need to implement some statistics way for showing stats.
"""
import os
import subprocess
import json
import time

input_directory = '/Users/geronimobasso/Desktop/extra/drones/database/originales-500'
dir_list = os.listdir(input_directory)

results = {}
yes_count = 0
no_count = 0


def generate_prompt(image_path):
    prompt = f"Does this image contain drones? Answer '1' if it does, or '0' if it does not: {image_path}."
    return prompt


def process_image(image_path):
    prompt = generate_prompt(image_path)
    command = f"ollama run llava prompt '{prompt}'"
    try:
        # Ejecutar el comando y capturar la salida
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return str(e)


start_time = time.time()

for filename in dir_list:
    _, file_extension = os.path.splitext(filename)
    if file_extension.lower() == '.jpg':  # Ensure case-insensitivity
        image_path = os.path.join(input_directory, filename)
        result = process_image(image_path)

        # Update results dictionary
        results[filename] = {"result": result}

        # Update counters
        if "1" in result.lower():
            yes_count += 1
        elif "0" in result.lower():
            no_count += 1

        print(f"{filename}: {result}")

# Add counters to the results
end_time = time.time()

elapsed_time = end_time - start_time

results['summary'] = {
    "yes_count": yes_count,
    "no_count": no_count
}

# Save results to a JSON file
output_file = 'output/detection_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Total processed images: {len(dir_list)}")
print(f"Results saved to {output_file}")
print(f"Yes count: {yes_count}, No count: {no_count}")
print(f"Time took for 500 images: {elapsed_time} seconds")
