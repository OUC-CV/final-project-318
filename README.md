# 计算机视觉项目报告

> 项目代码位于master分支，介绍视频链接：[B站视频链接地址](https://www.bilibili.com/video/BV1oigeekExc/?pop_share=1&vd_source=eef13c7caeb2529caa6bd395db286d7a)

## 一. 绪论

​	  在第一次中实验中，我们使用Debevec算法将多张曝光不同的图片合成为HDR图像，主要的过程为：图像对齐、恢复相机响应函数和使用MergeDebevec合成HDR图像，但实验效果不太理想。我们小组认为主要原因可能为图片对齐合成后会出现重影，以及使用的图像合成方法效果较差。为了实现更好的合成效果，我们从图像对齐和合成算法两方面入手查阅了相关文献，并使用已有的模型及算法对HDR图像重新合成。

​		针对图像对齐问题，我们采用了Phuoc-Hieu Le等人的论文提出的将单张图像生成多个曝光图像的模型。这种方式生成的图像对齐良好，不会有重影问题。对于合成方式，我们使用曝光融合算法对模型生成的多曝光图像重新合成。曝光融合算法的优势在于不需要标定相机响应曲线和曝光时间，并且跳过色调映射步骤，直接合成用于显示的效果图，不仅可以简化流程，还避免了色调映射过程中可能引入的伪影和失真问题，这有助于保持图像的自然外观和真实感。

​		为了评估生成的HDR图像的优劣，我们计算了一些评估指标来评估不同的合成方法。我们分别对第一次实验、Photomatix软件合成、曝光融合算法生成的HDR图像进行对比评估，使用的指标为：峰值信噪比PSNR、结构相似性SSIM 、感知图像补丁相似度LPIPS。通过指标的评估以及效果图可以看出，改进过后的HDR图像合成方式明显优于实验中算法结果。

## 二.  相关工作

### 1. 项目概况

​		HDR成像是当今图像领域不可缺少的技术。之前的方法中，侧重于对多幅曝光图像进行HDR重建，解决图像对齐、融合、色调映射等核心问题，但由于重建过程中出现了重影等问题，始终无法完美解决。目前的解决方案主要有：多重曝光融合、相机响应函数估计、卷积神经网络、端到端HDR合成图像等。HDR图像合成技术在不断发展，从传统的基于物理模型的方法到现代的深度学习方法，都在不断提高图像质量和处理效率。

​		对于图像效果较大的图像，我们可以肉眼分辨图像质量的好坏，但是对于细微的差别人眼无法精准的估计。对于图像质量的评估可以分为客观评估方法和主观评估方法，PSNR和SSIM等函数虽然量化了图像的质量，但是无法解释人类感知的差别。目前的图像质量评价多数采用预训练的模型和神经网络。LPIPS利用预训练的神经网络提取图像特征，然后计算特征之间的距离，而HDR-VDP-2是专门为HDR图像设计的质量评估指标，通过模拟人眼对亮度、对比度和视觉掩蔽效应的响应来评估图像质量。

​		在我们的项目中，我们首先完成了论文Single-Image HDR Reconstruction by Multi-Exposure Generation的复现，测试了论文中提出的模型性能，并且学习了其中单张照片生成多曝光图像的方法。为了更方便地进行图像处理，我们对测试文件的代码进行了修改并根据图片命名调整目录结构以便后续评估工作的进行。将生成的多曝光图使用Photomatix软件进行合成，得到HDR_pho图像，将多曝光图像使用曝光融合算法合成，得到HDR_exp图像。由于第一次实验的方法需要曝光时间，而生成的多曝光图像没有对应参数，所以仅对第一次实验中的图片使用三种合成方法生成HDR图像并比较。在评估时，第一次实验的每张原图具有三张不同方式合成的HDR图像，随机抽取的数据集每张原图具有两张同方式合成的HDR图像，我们对生成的图像用指标评估后可以评价HDR图像的合成效果。

### 2. 背景资料

#### 		2.1 图像对齐

参考论文：[Single-Image HDR Reconstruction by Multi-Exposure Generation](https://openaccess.thecvf.com/content/WACV2023/papers/Le_Single-Image_HDR_Reconstruction_by_Multi-Exposure_Generation_WACV_2023_paper.pdf)　　

​	我们使用了 Phuoc-Hieu Le等人提出的模型，该模型使用弱监督学习，通过反转相机响应，在合成多次曝光之前重建像素辐照度，将一张图片生成多个曝光图像，还能实现模拟填充欠曝和过曝区域的细节。将单张图像生成的多曝光图像不存在对齐的问题，在多曝光图像合成HDR图像之后几乎不会有重影。

#### 		2.2 HDR图像合成

参考论文：[Exposure Fusion](https://www.researchgate.net/publication/4295602_Exposure_Fusion/link/53f716940cf2888a7497691d/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19)

​	    这篇论文提出了新的HDR图像合成方法——曝光融合，即用一些指标衡量每张图像中像素的价值，通过拉普拉斯金字塔融合的方式得到HDR图像。实验过程中我们利用Phuoc-Hieu Le等人的论文中提出的模型生成的多曝光图像，使用曝光融合的方式对HDR图像进行合成。

#### 		2.3 结果评估

参考论文：[The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/pdf/1801.03924)

​	为对比项目生成的HDR图像的效果，我们计算了一些参数指标来评估不同的合成方法。峰值信噪比PSNR主要用于比较两幅图像之间的相似度或差异，PSNR大于30dB ，认为图像质量是好的。结构相似性SSIM 指标可以衡量图片的失真程度，SSIM更符合人眼的直观感受。SSIM取值在0-1之间，值越大质量越好。PSNR和SSIM两种函数无法解释人类感知的许多细微差别，感知图像补丁相似度LPIPS通过深度学习模型来评估两个图像之间的感知差异。LPIPS的值越低表示两张图像越相似，反之，则差异越大。

#### 		2.4 训练集和测试集

　　我们的训练集和测试集直接使用 Phuoc-Hieu Le的论文中使用的数据集[DrTMO](https://uithcm-my.sharepoint.com/:u:/g/personal/17520474_ms_uit_edu_vn/ET6uk6buZdlDnDkcJlRS_PEB6AoENYqFEqnPB5fn8r-oVQ?e=ddLdbw)。该数据集由Endo等人通过将Grossberg和Nayar的响应函数数据库(DoRF)中的5个代表性crf应用于收集的1043张具有9个曝光值的HDR图像而创建，该过程总共产生46,935张大小为512x512的用于训练的 LDR 图像和6210张用于测试的图像，其中涵盖了室内、室外、夜间和白天等各种场景。

### 3. 环境和工具

​		由于项目训练和测试模型对算力以及设备的硬件要求较高，我们使用了云服务器。该服务器搭载的GPU版本为 Tesla P40，实验中用到的显卡驱动为：Nvidia Driver 470.223.02， CUDA版本为： CUDA 11.3 ，cuDNN版本为：cudnn-linux-x86_64-8.9.6.50

本实验以下环境进行：

##### 操作系统

Ubuntu 18.04（模型测试，生成多曝光图像）

Windows 10（曝光融合，结果评估）

##### 编程语言

python 3.8

##### 依赖库

```python
# Ubuntu 18.04
PyTorch 1.11.0 + cu113
Torchvision 0.12.0+cu113
PyTorch Lighting 1.4.*
PIQ 
Albumentations
Pandas
tqdm
# Windows 10
OpenCV
Numpy
Matplotlib
Skimage
PyTorch
lpips
tqdm
```

## 三. 方法的具体细节

### 1. 单图像生成多曝光图像

​       Single-Image HDR Reconstruction by Multi-Exposure Generation论文中，提出了用于HDR图像重建的方法。基本思想是让网络学习从单张输入图像生成多个曝光图像，之后使用Photomatix软件来生成对应的HDR图像，色调映射之后显示图像。在我们的项目中，我们完成了对该论文的复现，并且将论文中提出的模型和使用的数据集应用于自己的项目。

#### 	1.1 原理解释

​      这种神经网络是一种新型的端到端的神经网络，是一种弱监督学习，可以生成任意数量的不同曝光图像，用于HDR图像重建。论文中提出的网络由两个阶段组成。第一个阶段是反向映射，在这个阶段中，使用HDR编码网络（N1）将输入图像Ii转换为Xi，这是一种适合图像在曝光时间Δti下的传感器曝光的潜在空间表示。通过适当的因子，可以根据公式：

![formula7](images/formula7.png)

对表示Xi进行缩放，从而获得在较短或较长曝光时间Δti±1下的传感器曝光。

​       接下来的阶段是前向映射，将场景辐照度Xi映射到改变曝光时间Δti后的像素值，以生成具有不同曝光的新图像。在这个阶段，需要在饱和区域中生成细节。由于欠曝光和过曝光图像的不同性质，我们提出使用两个子网络，分别是升曝光网络（N2）和降曝光网络（N3），分别生成与输入图像相关的高曝光和低曝光图像，这也遵循上述提出的公式。

![img](images/image1.jpg)

图片1：模型框架

#### 	1.2 模型的复现

> [原文代码github链接](https://github.com/VinAIResearch/single_image_hdr)
> 部分新增代码文件及功能：
> get_test_image.py -随机抽取100张测试图片
> single_test.py -对文件夹内所有图片生成多曝光图像
> test_output.py -根据原测试代码infer_backeted.py修改得到测试用例代码
> make_hdr.ipynb -生成hdr图像
> evaluate.py -评估

​		  由于训练集过大无法上传云服务器，导致无法从训练模型的步骤开始复现，我们采用了原文提供的预训练模型[pretrained.ckpt](https://uithcm-my.sharepoint.com/:u:/g/personal/17520474_ms_uit_edu_vn/EZa3EUzeLdNIibgD4vkixl4BgGTywlgSc9YnU7LRR4w_Jg?e=vgaYZr)，在配置环境后对其模型使用作者提供的测试代码进行测试验证预训练模型的性能。主要流程如下：

- 使用SSH工具MobaXterm连接远程服务器
- 由于我们购买的云服务器并没有安装显卡驱动，所以在配置环境之前首先安装了支持的版本的显卡驱动，CUDA，cuDNN

- 拉取原作者的代码库中的原代码

  ```python
  git clone https://github.com/VinAIResearch/single_image_hdr
  cd single_image_hdr
  ```

- 配置环境：安装Anaconda，根据配置文件[environment.yml](environment.yml)创建虚拟环境并安装对应版本的依赖库

- 原作者使用OneDrive上传的数据集和模型无法在远程终端直接下载，所以需要先将数据集和模型下载到本地，再使用STFP工具上传至云服务器。这里由于上传压缩包解压后储存空间不足，并且训练集过大，只能本地解压文件然后仅上传测试集数据。

- 根据文档提示设置匹配我们的环境的参数，修改部分测试文件代码以适应当前项目，运行测试脚本：

  ```python
  python test_output.py --out_dir results02/ \
      --ckpt pretrained.ckpt --in_name e.png \
      --num_workers 2 --test_dir training_data/Test/Bracketed_images/ \
      --test_label_path data/test_hdr.csv
  ```

- 根据`infer_bracketed.py`文件编写处理单个文件夹内所有图片的程序[single_test.py](single_test.py)

- 运行测试脚本：

  ```python
  # out_dir：输出结果所在的文件夹路径，输出结果目录结构为：single_image_hdr/out_dir/图片在测试集中的上级目录/image.png
  # img_path：需要处理的图片路径
  python single_test.py --out_dir output_test \
  	--ckpt pretrained.ckpt \
  	--img_dir /home/ubuntu/single_image_hdr/mytest
  ```

### 2. 曝光融合算法

​      曝光融合算法的主要思想是：通过一些具体指标，评估图像序列每个像素的价值，通过计算每张图每个像素的价值权重，通过加权融合的方式生成HDR图像。具体的权重指标有：对比度、饱和度、亮度。

​       对比度（Contrast）：对每张图像的灰度图应用拉普拉斯滤波器，并取滤波响应的绝对值。这样可以得到一个简单的对比度指标C。这个指标对于边缘和纹理等重要的信息分配很大的权重。

<img src="images/formula1.png" alt="image-20240621220518980" style="zoom: 80%;" />

​      饱和度（Saturation）：随着照片曝光时间的延长，高曝光的位置颜色会变得去饱和甚至没有颜色。饱和度指标S通过计算每个像素在R、G和B通道内的标准偏差来获得，可将RGB三个通道之间的标准差作为饱和度指标。

<img src="images/formula2.png" alt="image-20240621220709028" style="zoom: 50%;" />

​       曝光良好度（Well-exposedness）：观察一个通道内的原始强度值可以揭示像素的曝光情况。将取值在0.5左右的像素视为曝光良好，应该分配很大的权重；接近0和1的分别为欠曝和过曝应该分配很小的权重。像素值与其对应权重的关系符合均值为0.5的高斯分布：

<img src="images/formula3.png" alt="image-20240621220905368" style="zoom:80%;" />

​       在获取上述三个指标后，可以计算每个像素对应的权重，在融合时，需要对原始图像加权求和，得到HDR图像。合成图像时，使用了拉普拉斯金字塔融合的方式。从不同曝光的原始图像中分解出拉普拉斯金字塔，对应的权重图中分解出高斯金字塔，然后分别在每个尺度下进行融合，得到融合后的拉普拉斯金字塔。最后，从拉普拉斯金字塔的顶层开始向上采样，叠加同尺度的拉普拉斯细节，再向上采样和叠加细节，递归至最高分辨率，得到最终的结果。

![image](images/image2.png)

图片2：拉普拉斯图像融合的流程

### 3. HDR图像评估

​	    在评估过程中我们在测试集中随机抽取100张曝光度为1的图像并对其生成多曝光图像（正向曝光和反向曝光各四张），然后对生成的多曝光图像分别使用Photomatix软件，Exposure Fusion的多曝光合成hdr的方法以及第一次实验中使用的方法进行hdr合成。为对比项目生成的HDR图像的效果，我们计算了一些参数指标来评估不同的合成方法。具体使用的指标有为：PSNR、SSIM、LPIPS。下面对这三个指标进行介绍。

**峰值信噪比PSNR**

​	    峰值信噪比主要用于比较两幅图像之间的相似度或差异。PSNR是基于MSE(均方误差)定义的，对给定一个大小为m*n的原始图像I和对其添加噪声后的噪声图像K，其MSE可定义为：

<img src="images/formula4.jpg" alt="img" style="zoom: 67%;" />

则PSNR可定义为：

<img src="images/formula5.jpg" alt="img" style="zoom: 80%;" />

图像与影像压缩中典型的峰值讯噪比值在30dB 到50dB 之间，PSNR接近50dB ，代表压缩后的图像仅有些许非常小的误差。PSNR大于30dB ，人眼很难察觉压缩后和原始影像的差异，认为图像质量是好的。PSNR介于20dB 到30dB 之间，人眼就可以察觉出图像的差异，被认为图像质量不可接受。

**结构相似性SSIM**  

​		  SSIM 指标可以衡量图片的失真程度，也可以衡量两张图片的相似程度。与PSNR衡量绝对误差不同，SSIM更符合人眼的直观感受，是一种基于感知的模型。它将图像退化视为结构信息的感知变化，同时还结合了重要的感知现象，如亮度掩蔽和对比度掩蔽。SSIM值是通过不同的图像窗口计算的。 

<img src="images/formula6.jpg" alt="img" style="zoom: 67%;" />

SSIM取值在0-1之间，值越大质量越好。

**感知图像补丁相似度LPIPS**

​		  相较于LPIPS，PSNR和SSIM两种函数无法解释人类感知的许多细微差别。LPIPS通过深度学习模型来评估两个图像之间的感知差异。即使两个图像在像素级别上非常接近，人类观察者也可能将它们视为不同。因此，LPIPS 使用预训练的深度网络来提取图像特征，然后计算这些特征之间的距离，以评估图像之间的感知相似度。通常认为，LPIPS 比传统方法更符合人类的感知情况。LPIPS的值越低表示两张图像越相似，反之则差异越大。

## 四. 结果

#### 1. 第一次实验效果对比

> 使用实验中提供的数据集中曝光程度居中的图像作为生成多曝光图像的原图，分别为：exposures\img05.jpg，hdr\img_0.25.jpg，Memorial_SourceImages\memorial0066.png
>
> 从左到右依次为：原图，结果图，软件合成图，第一次实验结果图

img05.jpg：

<img src="images\img05.jpg" alt="img" style="zoom: 67%;" />

img_0.25.jpg：

<img src="images\img_0.25.jpg" alt="img" style="zoom: 67%;" />

memorial0066.png：

<img src="images\memorial0066.jpg" alt="img" style="zoom: 67%;" />

#### 2. 测试用例

​		除对实验中的图片对比外，我们还从数据集中随机抽取了100张图像，在生成多曝光图像后使用曝光融合算法以及软件合成HDR图像，并对结果进行分析评估以验证实验使用的算法性能。

评估结果如下，其中红色曲线为软件生成图评估结果，绿色曲线为曝光融合算法生成图评估结果：



<img src="images\evaluate.jpg" alt="img" style="zoom: 67%;" />



​		由于测试用例为随机取样，所以对于不同的图像使用相同方法合成的效果有较大差别，但对相同图像不同的方法得到的结果较为相似。观察数据可以看出，使用曝光融合算法生成的图像和软件生成HDR图像的效果相近，软件生成图的PSNR指标更好，但是差距不大。

效果图对比如下：

<img src="images\save.jpg" alt="img" style="zoom: 67%;" />

## 五.总结和讨论

​		在本项目中，我们针对HDR图像的合成效果进行了多方面的改进和分析，从结果中可以看出，经过我们改进之后，HDR图像的合成效果有了显著提升，基本解决了图像对齐不良和重影问题，增加了生成HDR图像的细节和真实感。在我们看来，通过深度学习模型生成的多曝光图像具有良好的对齐效果，避免了传统方法中多张图像对齐不良带来的重影问题。曝光融合算法不需要标定相机响应曲线，并且跳过色调映射步骤，直接生成可用于显示的HDR图像，简化了处理流程，减少了色调映射过程中可能引入的伪影和失真。

​		在完成项目的过程中，我们遇到了很多困难。在论文复现时，我们进行了很多尝试。刚开始我们尝试在自己的电脑windows系统的运行，但是有些函数和扩展包的版本不支持Windows系统，之后我们又尝试了虚拟机环境，但是无法解决GPU的问题，最后购买了云服务器才解决问题。在尝试使用模型时，论文中使用的数据集过大而云服务器的内存不足，无法一次性上传。在编写自己的测试用例时，刚开始使用模型生成的效果图效果很差，对于模型中的参数也无法精准使用，在经过多次的尝试后才生成了效果较好的HDR图像。

​		在解决问题的过程中，我们也有了一些新的反思和思考。我们的项目中使用曝光融合算法来合成HDR图像，这种方式跳过了色调映射步骤，而色调映射也是生成HDR图像中很重要的一个步骤，我们的项目中没有考虑这个因素。此外，在评估HDR图像时，我们使用了三个指标评价HDR图像的合成效果，虽然可以对HDR图像做出客观的评价，但HDR-VDP-2是专门为HDR图像设计的质量评估指标，我们的项目中并没有使用。

​		通过本次实验我们对于HDR图像的合成技术有了更详细全面的认识，学到了很多有关深度学习模型和云服务器工具的知识，对于我们的专业学习很很大帮助。

## 六.个人贡献声明

王锦瑞：搭建环境、代码编写、项目报告

郭雨菲：代码编写、项目报告、视频录制

## 七. 引用参考

[1] James A Ferwerda, Sumanta N Pattanaik, Peter Shirley, and Donald P Greenberg. A model of visual adaptation for realistic image synthesis. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pages 249–258, 1996.

[2] Xin Yang, Ke Xu, Yibing Song, Qiang Zhang, Xiaopeng Wei, and Rynson WH Lau. Image correction via deep reciprocating hdr transformation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1798–1807, 2018.

[3] P. Burt and T. Adelson. The Laplacian Pyramid as a Compact Image Code. IEEE Transactions on Communication, COM-31:532–540, 1983.

[4] L. Zhang, L. Zhang, X. Mou, and D. Zhang. Fsim: A feature similarity index for image quality assessment. TIP, 2011. 1, 2, 12, 14

[5]  A. Agrawal, R. Raskar, S. K. Nayar, and Y. Li. Removing photography artifacts using gradient projection and flashexposure sampling. ACM Trans. Graph., 24(3):828–835, 2005.
