import time
import os
import json
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

# Check for MPS availability
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Correct path based on the cache directory structure
start_time = time.time()
local_model_path = "/Users/geronimobasso/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/2e5297fcd60e2dd9b5a890a1421a93485921b277"
image_folder = "/Users/geronimobasso/Desktop/extra/drones/database/originales-500"

# Dictionary to store results
results = {}

# Process each image in the folder
for image_file in os.listdir(image_folder):
    if image_file.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        try:
            # Reinitialize the model and processor for each image
            model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path).to(device).eval()
            processor = AutoProcessor.from_pretrained(local_model_path)

            # Instruct the model to create a caption in Spanish
            prompt = "detect drones"
            model_inputs = processor(text=prompt, images=image, return_tensors="pt")

            # Move model inputs to the same device as the model
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            # Print the inputs to check for differences
            print(f"Model inputs for {image_file}: {model_inputs}")

            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

            # Check if the generation is empty
            if generation is None or len(generation) == 0:
                print(f"No generation output for {image_file}")
                continue

            # Decode the raw output
            result_full = processor.decode(generation[0], skip_special_tokens=True)

            # Extract the result part after the prompt
            result = result_full[len(prompt):].strip()

            # Check if result is empty after extraction
            if not result:
                print(f"Empty result after prompt extraction for {image_file}")
                continue

            # Store the result in the dictionary
            results[image_file] = {"result": result}

            # Debugging information
            print(result)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time for processing all images: {elapsed_time}")

# Save the results to a JSON file
with open("/Users/geronimobasso/Desktop/extra/drones/results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to results.json")