import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

"""
# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================
"""

#构造一维高斯核 window_size：表示高斯核的大小，即核的长度，sigma：标准差
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    #对高斯核进行归一化，确保其所有元素的和为 1,这样做的目的是确保输出图像的整体亮度不会因为卷积而改变
    return gauss / gauss.sum()


#生成一个用于 SSIM（结构相似性）计算的二维高斯窗口
def create_window(window_size, channel):#高斯窗口的大小和图像的通道数
    #在生成的一个一维的高斯核基础上，unsqueeze(1)在第 1 维（索引从 0 开始）增加一个维度，即[window_size] 变成 [window_size, 1]
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    #_1D_window.t())转置[1, window_size] 再进行矩阵相乘，得到[window_size, window_size] 的二维高斯核矩阵，最后增加两个维度，使其形状变为 [1, 1, window_size, window_size]。
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #expand使[1, 1, window_size, window_size] 扩展到 [channel, 1, window_size, window_size]
    #contiguous() 使得 window 在内存中连续存储，以便后续操作更高效 Variable(window)
    # 将 window 包装成一个 Variable 对象（这是旧版本 PyTorch 中常用的包装方式，目的是为了自动求导。
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    #通过高斯核对图像进行卷积，从而计算出每个像素点的局部均值 mu1 和 mu2
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    #方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    #衡量两幅图像 img1 和 img2 在局部区域内的相关性或协方差 反映了图像之间的结构相似性
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    #总体SSIM相似性包括luminance(亮度)、contrast(对比度)和structure(结构) ssim_map也是一个张量
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
        # 会对张量的所有维度进行平均，不指定任何维度时，它会将所有通道、所有像素的位置的 SSIM 值加在一起，然后计算整体的平均值。
        # 将 ssim_map 中所有的值看作一个一维向量，并计算这个向量的均值
    else:
        return ssim_map.mean(1).mean(1).mean(1)#计算每张图像的 SSIM 平均值

#计算两幅图像的局部方差 通过计算两幅图像的局部方差来评估它们的对比度
def Contrast(img1, img2, window_size=11, channel=1):
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    #将 window 张量的数据类型转换为与 img1 张量相同的数据类型。
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    return sigma1_sq, sigma2_sq#返回两个张量，分别表示 img1 和 img2 的局部方差


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average#是否对损失值取平均
        self.channel = 1#默认处理单通道图像
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        #判断当前通道数和窗口数据类型是否与之前相同
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            #如果输入图像 img1 在 GPU 上，则将窗口移动到同一设备
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# 和SSIMLoss类 功能一样
def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


# if __name__ == '__main__':
#     import cv2
#     from torch import optim
#     # from skimage import io
#
#     npImg1 = cv2.imread("einstein.png")
#
#     img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
#     img2 = torch.rand(img1.size())
#
#     if torch.cuda.is_available():
#         img1 = img1.cuda()
#         img2 = img2.cuda()
#
#     img1 = Variable(img1, requires_grad=False)
#     img2 = Variable(img2, requires_grad=True)
#
#     ssim_value = ssim(img1, img2).item()
#     print("Initial ssim:", ssim_value)
#
#     ssim_loss = SSIMLoss()
#     optimizer = optim.Adam([img2], lr=0.01)
#
#     while ssim_value < 0.99:
#         optimizer.zero_grad()
#         ssim_out = -ssim_loss(img1, img2)
#         ssim_value = -ssim_out.item()
#         print('{:<4.4f}'.format(ssim_value))
#         ssim_out.backward()
#         optimizer.step()
#     img = np.transpose(img2.detach().cpu().squeeze().float().numpy(), (1, 2, 0))
#     io.imshow(np.uint8(np.clip(img * 255, 0, 255)))
