import time
import os
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

input_directory = '/Users/geronimobasso/Desktop/extra/drones/database/originales-500'
dir_list = os.listdir(input_directory)
local_model_path = "/Users/geronimobasso/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/2e5297fcd60e2dd9b5a890a1421a93485921b277"

device = 'cpu'
print(f"Using device: {device}")

# Correct path based on the cache directory structure
star_time = time.time()

# Load the model and processor from the local path
model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path).eval()
processor = AutoProcessor.from_pretrained(local_model_path)

prompt = "detect drones"

for filename in dir_list:
    _, file_extension = os.path.splitext(filename)
    if file_extension.lower() == '.jpg':
        image_path = os.path.join(input_directory, filename)
        print(image_path)
        image = Image.open(image_path)

        model_inputs = processor(text=prompt, images=image, return_tensors="pt")

        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        print(processor.decode(generation[0],skip_special_tokens=True)[len(prompt):])

end_time = time.time()
elapsed_time = end_time - star_time

print(f"Total time for one image: {elapsed_time}")



