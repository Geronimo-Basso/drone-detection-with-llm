from huggingface_hub import notebook_login
import torch
import numpy as np
from PIL import Image
import requests
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch
import time
import os

notebook_login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = time.time()
image_folder = "/Users/geronimobasso/Desktop/extra/drones/database/originales-400"
results = {}

input_text = "Detect drone"
model_id = "google/paligemma-3b-pt-896"

for image_file in os.listdir(image_folder):
    if image_file.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        input_image = Image.open(image_path)

        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        processor = PaliGemmaProcessor.from_pretrained(model_id)

        inputs = processor(text=input_text, images=input_image,
                          padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
        model.to(device)
        inputs = inputs.to(dtype=model.dtype)

        with torch.no_grad():
          output = model.generate(**inputs, max_length=4200)

        print(processor.decode(output[0], skip_special_tokens=True))