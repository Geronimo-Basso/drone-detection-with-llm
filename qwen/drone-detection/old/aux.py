import os

def check_missing_images(directory):
    txt_count = 0
    jpg_count = 0
    missing_images = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_count += 1
                base_name = os.path.splitext(file)[0]
                txt_file_path = os.path.join(root, file)
                image_file_path = os.path.join(root, base_name + '.jpg')
                
                if not os.path.exists(image_file_path):
                    missing_images.append((txt_file_path, None))
                else:
                    missing_images.append((txt_file_path, image_file_path))
            elif file.endswith('.jpg'):
                jpg_count += 1

    return txt_count, jpg_count, missing_images


def has_more_than_one_line(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the first line
            first_line = file.readline()
            # Read the second line
            second_line = file.readline()
            # Check if there is a second line
            if second_line:
                return True
            else:
                return False
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return False




# Example usage
# directory = '/Users/geronimobasso/Desktop/extra/drones/database/Originales'
directory = 'qwen/inference-results/'
txt_count, jpg_count, missing_images = check_missing_images(directory)

print(f"Number of .txt files: {txt_count}")
print(f"Number of .jpg files: {jpg_count}")
print("Files and their corresponding images:")
for txt_file, img_file in missing_images:
#    if img_file:
#        print(f"{os.path.basename(txt_file)} -> {os.path.basename(img_file)}")
#    else:
#        print(f"{os.path.basename(txt_file)} -> No corresponding image found.")

    if has_more_than_one_line(txt_file):
        print(txt_file)
        print("The file has more than one line.")