import time
import os
import json
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

device = 'cpu'
print(f"Using device: {device}")

start_time = time.time()
local_model_path = "/Users/geronimobasso/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/2e5297fcd60e2dd9b5a890a1421a93485921b277"
image_folder = "/Users/geronimobasso/Desktop/extra/drones/database/originales-500"
results = {}

for image_file in os.listdir(image_folder):
    if image_file.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        try:
            model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path).eval()
            processor = AutoProcessor.from_pretrained(local_model_path)

            prompt = "detect drones"
            model_inputs = processor(text=prompt, images=image, return_tensors="pt")

            print(f"Model inputs for {image_file}")

            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

            if generation is None or len(generation) == 0:
                print(f"No generation output for {image_file}")
                continue

            result_full = processor.decode(generation[0], skip_special_tokens=True)

            result = result_full[len(prompt):].strip()

            if not result:
                print(f"Empty result after prompt extraction for {image_file}")
                continue

            results[image_file] = {"result": result}

            print(result)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time for processing all images: {elapsed_time}")

with open("/Users/geronimobasso/Desktop/extra/drones/results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to results.json")