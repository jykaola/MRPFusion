# MRPFusion



# 项目名称 (Project Name)

这是一个基于深度学习的红外与可见光图像融合方法研究的项目。该项目主要目的是构建轻量级的图像融合网络，并通过跨模态特征学习来提高图像融合的鲁棒性。以下为项目的简要说明以及如何安装和运行代码。

This is a project based on deep learning for infrared and visible image fusion. The goal of this project is to build a lightweight image fusion network and enhance the robustness of image fusion by learning cross-modal features. Below is a brief description of the project along with installation and usage instructions.

## 目录 (Table of Contents)

- [项目背景](#项目背景)
- [功能特性](#功能特性)
- [安装和运行](#安装和运行)
- [依赖库](#依赖库)
- [使用示例](#使用示例)
- [贡献](#贡献)
- [许可](#许可)

- [Background](#background)
- [Features](#features)
- [Installation and Usage](#installation-and-usage)
- [Dependencies](#dependencies)
- [Example Usage](#example-usage)
- [Contributing](#contributing)
- [License](#license)

## 项目背景 (Background)

随着红外与可见光图像融合技术的广泛应用，如何在保留图像细节的同时提高融合结果的质量和鲁棒性成为了一个重要问题。本项目提出了一种基于多尺度重参数化的轻量级融合网络，并通过跨模态特征学习提高图像融合的效果。

With the widespread application of infrared and visible image fusion techniques, improving the quality and robustness of fusion results while preserving image details has become an important issue. This project proposes a lightweight fusion network based on multi-scale re-parameterization and enhances the fusion results by learning cross-modal features.

## 功能特性 (Features)

- **轻量级设计**：采用轻量级网络架构，能够在性能与效率之间取得良好的平衡。
- **跨模态特征学习**：通过新的算法实现不同模态间的特征共享与融合。
- **高效的图像融合效果**：通过优化的模型在图像融合质量和计算效率上实现提升。

- **Lightweight Design**: Utilizes a lightweight network architecture that strikes a good balance between performance and efficiency.
- **Cross-modal Feature Learning**: Achieves feature sharing and fusion across different modalities through novel algorithms.
- **Efficient Image Fusion**: Optimizes the model to improve image fusion quality and computational efficiency.


建议使用 Conda 环境来管理依赖包。

It is recommended to use a Conda environment to manage dependencies.

```bash
conda create --name your_env_name python=3.8
conda activate your_env_name
```

### 3. 安装依赖库 (Install dependencies)

创建并激活环境后，安装项目所需的依赖：

```bash
pip install -r requirements.txt
```

### 4. 运行项目 (Run the project)

您可以使用以下命令运行项目：

You can run the project using the following command:

```bash
python train.py
python test.py
```

## 依赖库 (Dependencies)

- `pytorch` (根据您的实现选择深度学习框架)
- `opencv-python`
- `numpy`
- `matplotlib`
- `scikit-learn`

For detailed package dependencies, see `requirements.txt`.

# evalua_excel.py

## 概述 / Overview

`evalua_excel.py` 脚本用于计算并打印一系列图像质量评估指标。这些指标用于量化图像融合效果，帮助用户评估不同算法的性能。脚本接受三张图像：红外图像、可见光图像和融合图像，计算并保存包括以下指标：

The `evalua_excel.py` script is used to calculate and print a series of image quality evaluation metrics. These metrics are used to quantify the performance of image fusion algorithms, helping users evaluate the effectiveness of different algorithms. The script takes three images as input: infrared image, visible light image, and fused image, and calculates the following metrics:

- 熵（EN）/ Entropy (EN)
- 互信息（MI）/ Mutual Information (MI)
- 空间频率（SF）/ Spatial Frequency (SF)
- 平均梯度（AG）/ Average Gradient (AG)
- 标准差（SD）/ Standard Deviation (SD)
- 相关系数（CC）/ Correlation Coefficient (CC)
- 结构相似度（SCD）/ Structural Correlation Distance (SCD)
- 视觉信息保真度（VIF）/ Visual Information Fidelity (VIF)
- 均方误差（MSE）/ Mean Squared Error (MSE)
- 峰值信噪比（PSNR）/ Peak Signal-to-Noise Ratio (PSNR)
- Qabf
- Nabf
- SSIM
- MS_SSIM

## 结果输出 / Results Output

计算的结果将存储在一个 `DataFrame` 中，并最终输出到一个 Excel 文件中。该 Excel 文件包含不同算法的质量评估指标，方便用户进行比较和分析。

The computed results will be stored in a `DataFrame` and finally output to an Excel file. This Excel file contains the quality evaluation metrics for different algorithms, allowing users to compare and analyze the results.

## 使用方法 / How to Use

1. 将三张图像（红外图像、可见光图像和融合图像）作为输入。  
   Provide three images as input: infrared image, visible light image, and fused image.
   
2. 运行脚本，计算一系列图像质量指标。  
   Run the script to calculate a series of image quality metrics.
   
3. 最终结果会保存到 Excel 文件中，用户可以查看表格中的评估结果。  
   The final results will be saved to an Excel file, where users can view the evaluation results in the table.

## 依赖项 / Dependencies

- `pandas`
- `numpy`
- `openpyxl`
- `PIL`
- 其他图像处理相关库 / Other image processing libraries

## 备注 / Notes

- 脚本会将结果输出为 Excel 文件，文件名和路径可以根据需求修改。  
   The script outputs the results as an Excel file, and the file name and path can be modified as needed.

- 可根据需要调整输入图像和算法设置。  
   You can adjust the input images and algorithm settings as needed.



=======
# image-fusion
This is the code related to image fusion in the research paper.
>>>>>>> 51b33f5c43b599a38cafcadb5904d8c8baa9b232
