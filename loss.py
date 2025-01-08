# -*- encoding: utf-8 -*-
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_ssim import ssim #自定义模块，包含计算结构相似性（SSIM）损失的函数。SSIM用于衡量两幅图像在结构上的相似性，通常用于图像质量评估。
from my_util import bilateral_Filter#双边滤波
from scipy.signal import convolve2d



# sobel边缘算子
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)#核转换为 PyTorch 张量 unsqueeze(0) 是为了增加 batch 维度和通道维度。
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()  # 训练过程中不计算梯度，因为这个卷积核是固定的，不需要更新，所以设为 False。不可训练，为了保持网络完整性
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.Reflect = nn.ReflectionPad2d(1)#创建一个反射填充的层 1是填充的大小，即在每个边界周围添加 1 像素的填充

    def forward(self, x):
        x = self.Reflect(x)
        sobelx = F.conv2d(x, self.weightx, padding=0)
        sobely = F.conv2d(x, self.weighty, padding=0)
        return torch.abs(sobelx) + torch.abs(sobely)#计算水平和垂直方向的 Sobel 边缘图，并返回它们的绝对值之和，得到最终的边缘图。


#L_SSIM 类是一个自定义的损失函数，SSIM（结构相似性）指数，用于衡量图像之间的相似度。
#衡量融合图像与两个原始图像之间的不相似性的平均值如果 Loss_SSIM 值较大，说明融合图像与原始图像之间的相似性较低，融合质量可能不佳。
# 相反，如果 Loss_SSIM 值较小，说明融合图像与原始图像之间的相似性较高，融合质量较好。
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        # A = torch.mean(self.sobelconv(image_A))
        # B = torch.mean(self.sobelconv(image_B))
        # weight_A = A * A / (A * A + B * B)
        # weight_B = 1.0 - weight_A
        # #  计算权重，让纹理（梯度）更多的图有更高的权重
        # Loss_SSIM = (weight_A * (1 - ssim(image_A, image_fused)) + weight_B * (1 - ssim(image_B, image_fused))) * 0.5

        #SSIM（结构相似性指数）值的范围是 [0, 1]，其中 1 表示完全相似，0 表示完全不同。
        #ssim(image_A, image_fused)  表示计算 image_A 和 image_fused 之间的结构相似性指数，即相同程度
        #1−ssim，我们将相似性转化为差异性  值较小则差异越小
        Loss_SSIM = (1 - ssim(image_A, image_fused)) / 2 + (1 - ssim(image_B, image_fused)) / 2
        return Loss_SSIM


class Fusionloss(nn.Module):
    def __init__(self, weight=None):
        super(Fusionloss, self).__init__()
        if weight is None:
            weight = [10, 45, 0, 10]
        self.sobelconv = Sobelxy()
        '''
        双边滤波器是一种非线性滤波器，用于在平滑图像的同时保留边缘信息。与传统的均值滤波或高斯滤波不同，双边滤波器在平滑图像的过程中考虑了像素之间的颜色差异，
        这使得它能够有效地平滑噪声，但不会模糊图像的边缘。在图像融合任务中，使用双边滤波器的目的是在计算梯度前对图像进行预处理，从而减少噪声对梯度计算的干扰，
        确保梯度信息的准确性。
        '''
        #定义了一个双边滤波器 (bilateral_Filter) 对象，ksize=11滤波器的窗口大小或卷积核的大小，
        #sigmacolor=0.05即颜色空间中的标准差，控制颜色相似性的重要性。较小的值即颜色必须非常接近才能影响滤波器结果，较大的值则颜色差异较大的像素也会影响结果。
        #sigmaspace=8.0控制空间距离的重要性，决定了在图像中相隔多远的像素仍然能够互相影响。较大的值即使距离较远的像素也会相互影响，而较小的值只有相邻像素会有显著影响
        self.bila = bilateral_Filter(ksize=11, sigmacolor=0.05, sigmaspace=8.0)
        self.L_SSIM = L_SSIM()
        self.weight = weight
        # 初始化 alpha 为可学习参数，初始值为 0.5，可以根据需要调整初始值
        self.alpha = nn.Parameter(torch.tensor(0.5))  # alpha 是一个可学习的参数

    def forward(self, image_vis, image_ir, generate_img):#generate_img即融合后图像
        image_y = image_vis[:, :1, :, :]
        #公式中的减号和除以 𝐻𝑊这两个操作在 PyTorch 的 F.l1_loss 函数中已经隐式地处理了。
        #F.l1_loss(generate_img_grad, x_grad_joint) 计算的是 |generate_img_grad - x_grad_joint| 的均值，也就是每个元素的差的绝对值的均值。
        # 强度损失
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_in_max)

        ir_grad = self.sobelconv(self.bila(image_ir))  # 带双边滤波 先对红外图像 image_ir 进行双边滤波，再应用 Sobel 边缘检测算子
        # ir_grad = self.sobelconv(image_ir)
        y_grad = self.sobelconv(image_y)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        # 梯度损失
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)
        #loss_tradition 使用可学习的 alpha 计算 loss_tra
        alpha = torch.sigmoid(self.alpha)  # 通过Sigmoid函数限制alpha范围为[0, 1]
        loss_tra = F.l1_loss(generate_img, alpha * image_ir + (1 - alpha) * image_vis)
        # loss_tra = F.l1_loss(generate_img, (image_ir+image_y)*0.5)
        # SSIM损失
        loss_ssim = self.L_SSIM(image_y, image_ir, generate_img)
        # 总损失 loss_total=a*loss_in+b*loss_grad+c*loss_ssim+d+loss_tra
        loss_total = self.weight[0] * loss_in + self.weight[1] * loss_grad + \
                     self.weight[2] * loss_ssim + self.weight[3] * loss_tra
        # loss_total = self.weight[0][0] * loss_in + self.weight[0][1] * loss_grad + \
        #              self.weight[0][2] * loss_ssim + self.weight[0][3] * loss_tra
        return loss_total, loss_in, loss_grad, loss_ssim, loss_tra


if __name__ == '__main__':
    pass
