from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import os
import time

# Define the input directory and model path
input_directory = '/Users/geronimobasso/Desktop/extra/drones/database/originales-500'
local_model_path = "/Users/geronimobasso/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/2e5297fcd60e2dd9b5a890a1421a93485921b277"

# Initialize count and get the list of files in the directory
count = 0
dir_list = os.listdir(input_directory)

# Determine the device to use (CPU, GPU, MPS)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

print(f"Using device: {device}")

# Record the start time
start_time = time.time()

# Load the model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path).eval().to(device)
processor = AutoProcessor.from_pretrained(local_model_path)
prompt = "detect drones en"

print("Finished loading model")
print("Starting with the predictions")

# Loop through each file in the directory
for filename in dir_list:
    _, file_extension = os.path.splitext(filename)
    if file_extension.lower() == '.jpg':
        count += 1
        filename_complete = os.path.join(input_directory, filename)
        print(f"Start prediction {count} with {filename_complete}")

        try:
            image = Image.open(filename_complete).convert("RGB")
            model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**model_inputs, max_new_tokens=250, do_sample=False)
                generation = generation[0][input_len:]
                decoded = processor.decode(generation, skip_special_tokens=True)
                print(decoded)
        except Exception as e:
            print(f"Failed to process {filename_complete}: {e}")

# Record the end time and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time {elapsed_time}")