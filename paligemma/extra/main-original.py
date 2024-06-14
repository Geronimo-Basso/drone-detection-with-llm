# from PIL import Image, ImageDraw
# import re
#
# # Load the image
# image_path = '/Users/geronimobasso/Desktop/extra/drones/database/originales-2k/img1032273.jpg'
# image = Image.open(image_path)
#
# # Dimensions of the image
# width, height = image.size
#
# # Coordinates provided by the LLM model
# coordinates_str = '<loc0436><loc0646><loc0593><loc1022>'
#
# # Extract numerical values from the coordinates string
# coords = re.findall(r'\d+', coordinates_str)
# x_min, y_min, x_max, y_max = map(int, coords)
#
# # Normalize the coordinates (assuming they are given in range 0-1000)
# x_min = x_min / 1000 * width
# y_min = y_min / 1000 * height
# x_max = x_max / 1000 * width
# y_max = y_max / 1000 * height
#
# # Draw the bounding box
# draw = ImageDraw.Draw(image)
# draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
#
# # Save the modified image
# output_path = '/Users/geronimobasso/Desktop/bounded_img1004233.jpg'
# image.save(output_path)
#
# output_path

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

print(f"Using device: {device}")

# Correct path based on the cache directory structure
local_model_path = "/Users/geronimobasso/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/2e5297fcd60e2dd9b5a890a1421a93485921b277"

image = Image.open("/Users/geronimobasso/Desktop/extra/drones/database/originales-2k/img1032273.jpg")

# Load the model and processor from the local path
model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path).eval().to(device)
processor = AutoProcessor.from_pretrained(local_model_path)

# Instruct the model to create a caption in Spanish
prompt = "detect drones en"
model_inputs = processor(text=prompt, images=image, return_tensors="pt")

if device == 'mps':
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}  # Move inputs to the correct device

input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
