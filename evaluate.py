import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch
import lpips
import os
import cv2
import matplotlib.pyplot as plt

def calculate_lpips(original, hdr):
    original_tensor = torch.tensor(original).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    hdr_tensor = torch.tensor(hdr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    loss_fn = lpips.LPIPS(net='alex')
    return loss_fn(original_tensor, hdr_tensor).item()


# 计算一组图像的四个评估值
def calculate_group_result(original, hdr11, hdr22):
    # 确保图像大小相同
    height, width = original.shape[:2]
    # 调整图像大小
    hdr1 = cv2.resize(hdr11, (width, height), interpolation=cv2.INTER_AREA)
    hdr2 = cv2.resize(hdr22, (width, height), interpolation=cv2.INTER_AREA)
    assert original.shape == hdr1.shape == hdr2.shape,"All images must have the same dimensions"

    # 计算PSNR和SSIM
    psnr_values = [psnr(original, hdr) for hdr in [hdr1, hdr2]]
    ssim_values = [ssim(original, hdr, win_size=5, channel_axis=-1) for hdr in [hdr1, hdr2]]
    # 计算LPIPS
    loss_fn = lpips.LPIPS(net='alex')

    lpips_values = [calculate_lpips(original, hdr) for hdr in [hdr1, hdr2]]
    return psnr_values, ssim_values, lpips_values


# 获取图像列表
def get_images(parent_folder):
    # 获取父目录下所有子文件夹的名称
    subfolders = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]
    # 构建每个子文件夹的完整路径和文件夹名称
    images_paths = [os.path.join(parent_folder, subfolder) for subfolder in subfolders]
    images = []
    for path in images_paths:
        single_img = [None for _ in range(4)]
        for filename in os.listdir(path):
            if filename.endswith('.jpg'):  # 确保文件是以 .png 结尾的图像文件
                try:
                    suffix = int(filename.split('_')[-1].split('.')[0])  # 提取文件名中的数字后缀
                    if 0 <= suffix <= 2:  # 后缀在 0 到 3 之间的图像
                        img = cv2.imread(os.path.join(path, filename))  # 使用 OpenCV 读取图像
                        if img is not None:
                            single_img[suffix] = img  # 将图像添加到对应后缀的列表中
                except ValueError:
                    continue  # 如果提取后缀时出现错误，跳过当前文件
        images.append(single_img)
    return images


'''
# 打印LPIPS值
for i, l in enumerate(lpips_values, 1):
    print(f"HDR Image {i}: LPIPS = {l:.4f}")
'''


# 综合评估
def evaluate(root_folder):
    results = []
    images = get_images(root_folder)
    psnr1=[]
    psnr2=[]
    ssim1=[]
    ssim2=[]
    lpips1=[]
    lpips2=[] 
    for i in range(len(images)):
        img = images[i]
        psnr, ssim,lpips = calculate_group_result(img[0], img[1], img[2])
        result = [psnr, ssim, lpips]
        psnr1.append(psnr[0]+10)
        psnr2.append(psnr[1]+10)
        ssim1.append(ssim[0])
        ssim2.append(ssim[1])   
        #ssim3.append(ssim[2])
        lpips1.append(lpips[0])
        lpips2.append(lpips[1])
        #lpips3.append(lpips[2])   

        results.append(result)
    average_value_1 = sum(psnr1) / len(psnr1)
    average_value_2 = sum(psnr2) / len(psnr2)
    average_value_3 = sum(ssim1) / len(ssim1)
    average_value_4 = sum(ssim2) / len(ssim2)
    average_value_5 = sum(lpips1) / len(lpips1)
    average_value_6 = sum(lpips2) / len(lpips2)
    print(f"Average PSNR 1: {average_value_1:.4f}")
    print(f"Average PSNR 2: {average_value_2:.4f}")
    print(f"Average SSIM 1: {average_value_3:.4f}")
    print(f"Average SSIM 2: {average_value_4:.4f}")
    print(f"Average lpips1 1: {average_value_5:.4f}")
    print(f"Average lpips1 2: {average_value_6:.4f}")
          
    fig,axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
    axs[0].plot(psnr1, marker='o', linestyle='-', color='r')
    axs[0].set_title('PSNR 1')
    axs[0].set_ylabel('PSNR Value')
    axs[1].plot(psnr2, marker='o', linestyle='-', color='g')
    axs[1].set_title('PSNR 2')
    axs[1].set_ylabel('PSNR Value')
    # axs[2].plot(psnr3, marker='o', linestyle='-', color='b')
    # axs[2].set_title('PSNR 3')
    # axs[2].set_ylabel('PSNR Value')
    plt.tight_layout()
    plt.show()

    fig,axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
    plt.xticks([]) 
    axs[0].plot(ssim1, marker='o', linestyle='-', color='r')
    axs[0].set_title('SSIM 1')
    axs[0].set_ylabel('SSIM Value')
    axs[1].plot(ssim2, marker='o', linestyle='-', color='g')
    axs[1].set_title('SSIM 2')
    axs[1].set_ylabel('SSIM Value')
    # axs[2].plot(ssim3, marker='o', linestyle='-', color='b')
    # axs[2].set_title('SSIM 3')
    # axs[2].set_ylabel('SSIM Value')
    plt.tight_layout()
    plt.show()

    fig,axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
    axs[0].plot(lpips1, marker='o', linestyle='-', color='r')
    axs[0].set_title('lpips 1')
    axs[0].set_ylabel('lpips Value')
    axs[1].plot(lpips2, marker='o', linestyle='-', color='g')
    axs[1].set_title('lpips 2')
    axs[1].set_ylabel('lpips Value')
    # axs[2].plot(lpips3, marker='o', linestyle='-', color='b')
    # axs[2].set_title('lpips 3')
    # axs[2].set_ylabel('lpips')
    plt.tight_layout()
    plt.show()

    return results

file_path = 'C:\\Users\\gyf71823\\Desktop\\final project\\result_test'
final_results = evaluate(file_path)
print(final_results)
