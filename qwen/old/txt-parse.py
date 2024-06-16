import os
import shutil

def copy_txt_files(src_folder, dest_folder):
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Loop through all files in the source folder
    for filename in os.listdir(src_folder):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            # Construct the full file path
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(dest_folder, filename)
            # Copy the file to the destination folder
            shutil.copy2(src_file, dest_file)
            print(f'Copied: {src_file} to {dest_file}')

# Define source and destination folders
src_folder = 'qwen/inference-results/'


dest_folder = 'qwen/inferece-results-2'

# Call the function to copy .txt files
copy_txt_files(src_folder, dest_folder)