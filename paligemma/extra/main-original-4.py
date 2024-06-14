import time
import os
import json
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

device = 'cpu'
print(f"Using device: {device}")

# Correct path based on the cache directory structure
start_time = time.time()
local_model_path = "/Users/geronimobasso/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/2e5297fcd60e2dd9b5a890a1421a93485921b277"
image_folder = "/Users/geronimobasso/Desktop/extra/drones/database/originales-500"

# Load the model and processor from the local path
model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path).eval()
processor = AutoProcessor.from_pretrained(local_model_path)

# Dictionary to store results
results = {}

# Process each image in the folder
for image_file in os.listdir(image_folder):
    if image_file.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        # Instruct the model to create a caption in Spanish
        prompt = "detect drones"
        model_inputs = processor(text=prompt, images=image, return_tensors="pt")

        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        result = processor.decode(generation[0], skip_special_tokens=True)[len(prompt):]

        print(processor.decode(generation[0], skip_special_tokens=True)[len(prompt):])

        # Store the result in the dictionary
        results[image_file] = {"result": result}

        print(result)
        print(f"Processed {image_file}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time for processing all images: {elapsed_time}")

# Save the results to a JSON file
with open("/Users/geronimobasso/Desktop/extra/drones/results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to results.json")