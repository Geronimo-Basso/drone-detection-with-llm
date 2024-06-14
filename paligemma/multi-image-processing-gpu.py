import time
import os
import json
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

# Check for MPS availability
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Correct path based on the cache directory structure
start_time = time.time()
local_model_path = "/Users/geronimobasso/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/2e5297fcd60e2dd9b5a890a1421a93485921b277"
image_folder = "/Users/geronimobasso/Desktop/extra/drones/database/originales-500"

# Load the model and processor once
model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path).to(device).eval()
processor = AutoProcessor.from_pretrained(local_model_path)

# Dictionary to store results
results = {}

# Process each image in the folder
for image_file in os.listdir(image_folder):
    if image_file.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        try:
            # Prompt for the model
            prompt = "detect drones"
            model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

            # Generate output
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

            # Decode the raw output
            result_full = processor.decode(generation[0], skip_special_tokens=True)

            # Extract the result part after the prompt
            result = result_full[len(prompt):].strip()

            # Store the result in the dictionary
            results[image_file] = {"result": result}

            # Print debugging information
            print(f"Processed {image_file} with result: {result}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time for processing all images: {elapsed_time}")

# Save the results to a JSON file
with open("/Users/geronimobasso/Desktop/extra/drones/results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to results.json")