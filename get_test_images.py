import os
import shutil
import random

# 原始目录路径
base_dir = '/home/ubuntu/single_image_hdr/training_data/Test/Bracketed_images'

# 目标文件夹路径
destination_folder = '/home/ubuntu/single_image_hdr/myTest'
os.makedirs(destination_folder, exist_ok=True)

# 查找所有子目录中的e.png文件
all_e_png_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == 'e.png':
            all_e_png_files.append(os.path.join(root, file))

# 随机抽取100份e.png文件
if len(all_e_png_files) < 100:
    print("Error: There are fewer than 100 'e.png' files available.")
else:
    selected_files = random.sample(all_e_png_files, 100)

    # 复制文件到目标文件夹
    for file_path in selected_files:
        # 提取上级目录名作为文件名的一部分，防止重名冲突
        parent_dir_name = os.path.basename(os.path.dirname(file_path))
        new_file_name = f"{parent_dir_name}_e.png"
        new_file_path = os.path.join(destination_folder, new_file_name)
        shutil.copyfile(file_path, new_file_path)

    print(f"Successfully copied 100 'e.png' files to {destination_folder}")
