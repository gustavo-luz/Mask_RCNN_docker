import os
import random
import shutil

folder_path = "test/"

all_files = os.listdir(folder_path)

#print(all_files)

num_files_to_select = int(0.03 * len(all_files))
# Randomly select files
selected_files = random.sample(all_files, num_files_to_select)
output_folder = "test_cut"
os.makedirs(output_folder, exist_ok=True)

# Copy the selected files to the output folder
for file_name in selected_files:
    src_file = os.path.join(folder_path, file_name)
    dst_file = os.path.join(output_folder, file_name)
    shutil.copy(src_file, dst_file)

print(f"Selected {num_files_to_select} random photos and copied them to {output_folder}")
