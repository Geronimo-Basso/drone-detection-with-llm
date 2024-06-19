import os

# Load all text files
base_directory = '/Users/geronimobasso/Desktop/extra/drones/code/computer-vision/originales-400-txt-edit'
txt_filenames = []

with os.scandir(base_directory) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.lower().endswith('.txt'):
            txt_filenames.append(entry.path)

print(f"Text files in total: {len(txt_filenames)}")


# Function to refactor the text file
def refactor_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 1:
            modified_line = " ".join(parts[1:])
            modified_lines.append(modified_line)

    with open(file_path, 'w') as file:
        file.write("\n".join(modified_lines) + "\n")


# Processing text files
for txt_path in txt_filenames:
    refactor_text_file(txt_path)

print(f"Refactored {len(txt_filenames)} text files")