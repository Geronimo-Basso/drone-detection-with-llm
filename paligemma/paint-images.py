from PIL import Image, ImageDraw
import re

# Load the image
image_path = '/Users/geronimobasso/Desktop/extra/drones/database/Originales/img1040213.jpg'
image = Image.open(image_path)

# Dimensions of the image
width, height = image.size

# Coordinates provided by the LLM model
coordinates_str = '<loc0412><loc0318><loc0574><loc0368>'

# Extract numerical values from the coordinates string
coords = re.findall(r'\d+', coordinates_str)
x_min, y_min, x_max, y_max = map(int, coords)

# Normalize the coordinates (assuming they are given in range 0-1000)
x_min = x_min / 1000 * width
y_min = y_min / 1000 * height
x_max = x_max / 1000 * width
y_max = y_max / 1000 * height

# Draw the bounding box
draw = ImageDraw.Draw(image)
draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)

# Save the modified image
output_path = '/Users/geronimobasso/Desktop/img1040213.jpg'
image.save(output_path)