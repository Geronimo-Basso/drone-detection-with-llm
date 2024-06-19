import os
from pathlib import Path
from PIL import Image, ImageDraw

# Load all images and text files
base_directory = '/Users/geronimobasso/Desktop/extra/drones/code/computer-vision/originales-400'
output_directory = '/Users/geronimobasso/Desktop/extra/drones/code/computer-vision/originales-400-bb'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

image_count = 0
images_filenames = []
txt_filenames = []

with os.scandir(base_directory) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.lower().endswith('.jpg'):
            image_count += 1
            images_filenames.append(entry.path)
        elif entry.is_file() and entry.name.lower().endswith('.txt'):
            txt_filenames.append(entry.path)

print(f"Images in total: {len(images_filenames)}")
print(f"Text files in total: {len(txt_filenames)}")

# Function to parse bounding boxes from text files
def parse_bounding_boxes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    bounding_boxes = []
    for line in lines:
        if line.strip() != "0 0 0 0":
            coords = list(map(float, line.strip().split()[1:]))  # Ignore the leading '0'
            bounding_boxes.append(coords)
    return bounding_boxes

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, bounding_boxes):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in bounding_boxes:
        center_x, center_y, box_width, box_height = box
        left = (center_x - box_width / 2) * width
        right = (center_x + box_width / 2) * width
        top = (center_y - box_height / 2) * height
        bottom = (center_y + box_height / 2) * height
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
    return image

# Processing images
for file_name in images_filenames:
    image_path = file_name
    txt_path = file_name.replace('.jpg', '.txt')

    if not os.path.exists(txt_path):
        continue

    bounding_boxes = parse_bounding_boxes(txt_path)
    with Image.open(image_path) as img:
        img_with_boxes = draw_bounding_boxes(img, bounding_boxes)
        output_path = os.path.join(output_directory, Path(file_name).name)
        img_with_boxes.save(output_path)

print(f"Processed {len(images_filenames)} images and saved them to {output_directory}")