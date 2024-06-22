import os

base_dir = 'myOutput'
output_dir = 'result'
batch_file = '/home/ubuntu/single_image_hdr/batch_list.txt'

# 获取所有子目录
subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

with open(batch_file, 'w') as f:
    for subdir in subdirs:
        output_path = os.path.join(output_dir, os.path.basename(subdir) + '_2.tif')
        f.write(f"-merge -src {subdir} -out {output_path}\n")

print(f"Batch file written to {batch_file}")
