import logging
import os
import numpy as np
import torch
from PIL import Image
import warnings
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from natsort import natsorted
from scipy.signal import convolve2d
import math
from scipy.fftpack import dctn
from scipy.ndimage import sobel, generic_gradient_magnitude
from torch import nn
from prettytable import PrettyTable
from tqdm import tqdm
from logger import setup_logger


warnings.filterwarnings("ignore")#忽略所有的警告信息

#计算一系列图像质量评估指标
'''
ir_name: 红外图像文件名
vi_name: 可见光图像文件名
f_name: 融合图像文件名
easy_flag: 布尔值，默认为 False，如果为 True 则会简化一些计算
'''
def evaluation_one(ir_name, vi_name, f_name, easy_flag=False):
    f_img = Image.open(f_name).convert('L')#打开图像转为单通道
    ir_img = Image.open(ir_name).convert('L')
    vi_img = Image.open(vi_name).convert('L')
    f_img_int = np.array(f_img).astype(np.int32)
    # 转换为 NumPy 数组
    f_img_double = np.array(f_img).astype(np.float32)
    ir_img_int = np.array(ir_img).astype(np.int32)
    ir_img_double = np.array(ir_img).astype(np.float32)

    vi_img_int = np.array(vi_img).astype(np.int32)
    vi_img_double = np.array(vi_img).astype(np.float32)

    EN = EN_function(f_img_int)
    MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)
    SF = SF_function(f_img_double)
    SD = SD_function(f_img_double)
    AG = AG_function(f_img_double)
    # PSNR = PSNR_function(ir_img_double, vi_img_double, f_img_double)
    # MSE = MSE_function(ir_img_double, vi_img_double, f_img_double)
    VIF = VIF_function(ir_img_double, vi_img_double, f_img_double)
    # CC = CC_function(ir_img_double, vi_img_double, f_img_double)
    SCD = SCD_function(ir_img_double, vi_img_double, f_img_double)
    # Qabf = Qabf_function(ir_img_double, vi_img_double, f_img_double)
    if easy_flag:
        Nabf, SSIM, MS_SSIM = 0.0, 0.0, 0.0
    else:
        Nabf = Nabf_function(ir_img_double, vi_img_double, f_img_double)
        SSIM = SSIM_function(ir_img_double, vi_img_double, f_img_double)
        MS_SSIM = MS_SSIM_function(ir_img_double, vi_img_double, f_img_double)

    # return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM
    return EN, MI, SF, AG, SD, VIF,SCD
#用于计算输入图像的熵（Entropy）
#image_array表示图像的 NumPy 数组，通常是 2D 灰度图像
def EN_function(image_array):
    # 计算图像的直方图
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 255))
    # 将直方图归一化
    histogram = histogram / float(np.sum(histogram))
    # 计算熵
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy

#计算出的图像空间频率 SF
def SF_function(image):
    image_array = np.array(image)
    RF = np.diff(image_array, axis=0)#计算行频率 计算图像中相邻行之间的差值，结果是一个比原图少一行的数组
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(image_array, axis=1)#计算列频率 (CF)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)#总空间频率 (SF)
    return SF

#计算图像标准差
def SD_function(image_array):
    m, n = image_array.shape#高度、宽度
    u = np.mean(image_array)#计算图像中所有像素的均值 即图像的平均亮度或灰度值
    SD = np.sqrt(np.sum(np.sum((image_array - u) ** 2)) / (m * n))#计算标准差
    #(image_array - u)是计算每个像素值与均值 u 的差异
    #(image_array - u) ** 2 计算每个差异的平方，这一步是为了消除差异的符号（正负值）
    return SD

# 计算图像之间峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）
# A、B 和 F 分别表示三张灰度图像的像素矩阵
def PSNR_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0# 归一化到 0 到 1 的范围内
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)# A 和 F 之间的均方误差 MSE
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)# B 和 F 之间的 MSE
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * np.log10(255 / np.sqrt(MSE))# PSNR = 20 * np.log10(255 / np.sqrt(MSE))
    return PSNR

#计算图像之间均方误差（Mean Squared Error, MSE）
def MSE_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE

#生成 2D 高斯滤波器（Gaussian filter）掩膜的函数
def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]#计算高斯滤波器掩膜的中心坐标 eg:shape=(5,5) ss=(5,5) m,n=(2,2)
    y, x = np.ogrid[-m:m + 1, -n:n + 1]#创建坐标网格-m~m,-n~n
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))# 计算高斯掩码 是一个矩阵
    h[h < np.finfo(h.dtype).eps * h.max()] = 0#高斯滤波器掩膜 h 中非常小的元素值设置为 0，以提高计算的稳定性并减少数值误差。
    sumh = h.sum()
    if sumh != 0:
        h /= sumh#归一化
    return h

#计算视觉信息保真度 (VIF) 指标的尺度加权版本
# ref: 参考图像，通常是原始图像或质量较好的图像。
# dist: 失真图像，通常是经过处理或压缩后的图像。
def vifp_mscale(ref, dist):
    sigma_nsq = 2#噪声方差的常量
    num = 0#VIF 的分子
    den = 0#分母
    for scale in range(1, 5):#变量控制不同的尺度层次
        N = 2 ** (4 - scale + 1) + 1#计算当前尺度下的高斯滤波器大小
        win = fspecial_gaussian((N, N), N / 5)#是当前尺度下的高斯滤波器

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')#通过高斯滤波器对图像进行平滑处理
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]#下采样 ref[::2, ::2] 表示从 ref 图像中每隔一个像素选择一个像素
            dist = dist[::2, ::2]
        #计算了图像对比度、均值和方差等统计量
        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        # 计算均值的平方和乘积
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        #方差 协方差
        sigma1_sq = convolve2d(ref * ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist * dist, win, mode='valid') - mu2_sq
        #协方差
        sigma12 = convolve2d(ref * dist, win, mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        #局部回归系数 g 和局部残差方差 sv_sq
        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        #修正异常值
        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10


        num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))#分子
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))#分母
    vifp = num / den#vif
    return vifp


def VIF_function(A, B, F):
    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF

#计算了图像质量评价中的 互相关系数
def CC_function(A, B, F):
    rAF = np.sum((A - np.mean(A)) * (F - np.mean(F))) / np.sqrt(
        np.sum((A - np.mean(A)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    rBF = np.sum((B - np.mean(B)) * (F - np.mean(F))) / np.sqrt(
        np.sum((B - np.mean(B)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    CC = np.mean([rAF, rBF])
    return CC

#计算了两个图像或矩阵的 皮尔逊相关系数 范围在 [-1, 1] 之间，1 表示完全正相关，-1 表示完全负相关，0 表示没有线性相关
def corr2(a, b):
    #去均值化 衡量两个变量之间的线性关系
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r


def SCD_function(A, B, F):
    r = corr2(F - B, A) + corr2(F - A, B)
    #r 值较高，说明图像 F 相对于参考图像 A 和 B 的结构保持较好，失真程度较低。
    #r 值较低，则说明图像 F 在结构上与 A 和 B 之间的差异较大，失真程度较高。
    return r

#get_Qabf 的直接封装
def Qabf_function(A, B, F):
    return get_Qabf(A, B, F)


def Nabf_function(A, B, F):
    return Nabf_function(A, B, F)

#计算两个图像 im1 和 im2 之间的互信息,互信息是一种用于衡量两个变量之间相互依赖性的量度，在图像处理中常用于图像对齐、配准等任务。
#im1 和 im2：两个输入的灰度图像 gray_level：灰度级别的数量
def Hab(im1, im2, gray_level):
    hang, lie = im1.shape
    count = hang * lie
    N = gray_level
    h = np.zeros((N, N))
    for i in range(hang):
        for j in range(lie):
            h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1#举例 h[0,1] = 1 表示在图像中 (im1[i,j] = 0, im2[i,j] = 1) 的组合出现了1次。
    h = h / np.sum(h)#归一化联合直方图
    #计算边缘概率分布
    im1_marg = np.sum(h, axis=0)
    im2_marg = np.sum(h, axis=1)
    #计算熵 (Entropy)
    H_x = 0#im1的熵
    H_y = 0#im2的熵
    for i in range(N):
        if (im1_marg[i] != 0):
            H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])
    for i in range(N):
        if (im2_marg[i] != 0):
            H_x = H_x + im2_marg[i] * math.log2(im2_marg[i])
    #计算联合熵
    H_xy = 0
    for i in range(N):
        for j in range(N):
            if (h[i, j] != 0):
                H_xy = H_xy + h[i, j] * math.log2(h[i, j])
    #计算互信息
    MI = H_xy - H_x - H_y
    return MI

#两个输入图像 A 和 B 相对于目标图像 F 的互信息
def MI_function(A, B, F, gray_level=256):
    MIA = Hab(A, F, gray_level)
    MIB = Hab(B, F, gray_level)
    MI_results = MIA + MIB
    return MI_results

'''
平均梯度值高的图像通常意味着有更多的细节和边缘信息，图像看起来更清晰；相反，低的平均梯度值则意味着图像更加平滑或模糊。
这个指标的意义在于提供了一种定量的方式来评估图像的清晰度和细节丰富程度。
假设我们有两张图像：
图像 A 是一张清晰的边缘分明的图片，例如一张建筑物的照片。
图像 B 是一张模糊的图片，例如在拍摄时相机抖动造成的模糊图像。
对于图像 A，其边缘处的像素值变化剧烈，因此梯度较大，导致平均梯度较高；而图像 B 由于模糊，像素值变化平缓，梯度较小，导致平均梯度较低。
'''
#计算一幅图像的平均梯度（Average Gradient, AG）
def AG_function(image):
    width = image.shape[1]
    width = width - 1
    height = image.shape[0]
    height = height - 1# 梯度计算后，边缘的像素点会减少1
    tmp = 0.0
    [grady, gradx] = np.gradient(image)#x 方向和 y 方向的梯度
    s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)#每个像素点的实际梯度幅度
    AG = np.sum(np.sum(s)) / (width * height)#梯度幅度的平均值
    return AG


#计算了图像A和图像B与参考图像F之间的结构相似性（SSIM）
def SSIM_function(A, B, F):
    ssim_A = ssim(A, F)
    ssim_B = ssim(B, F)
    SSIM = 1 * ssim_A + 1 * ssim_B
    return SSIM.item()


def MS_SSIM_function(A, B, F):
    ssim_A = ms_ssim(A, F)
    ssim_B = ms_ssim(B, F)
    MS_SSIM = 1 * ssim_A + 1 * ssim_B
    return MS_SSIM.item()


def Nabf_function(A, B, F):
    Nabf = get_Nabf(A, B, F)
    return Nabf


#生成一个一维的高斯滤波器
def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    #size窗口的大小 sigma高斯函数的标准差
    coords = torch.arange(size, dtype=torch.float32)#创建一个包含 size 个元素的张量，值从 0 到 size-1。
    coords -= size // 2#将这些值中心化，使得中心点为 0

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)#扩展通道为[1, 1, size]

#用于应用一维高斯卷积核对输入张量进行模糊处理,适用于不同维度的输入张量（例如 4D 和 5D 张量），并且对每个维度应用高斯卷积。
def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    #input (torch.Tensor): 这是要进行模糊处理的张量，形状可以是 [B, C, H, W] 或 [B, C, D, H, W]
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    #检查除了第一个维度和最后一个维度之外的所有维度是否都是1，即以win=[3, 1, 5]为例， 表示取第二个到倒数第二个 ，win.shape[1:-1]=1即高度
    #，win.shape 作为错误信息的一部分显示出来。
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]#输入张量的通道数 C
    out = input#初始化 out 为输入张量
    #i: 维度的索引 s：当前维度的实际大小
    for i, s in enumerate(input.shape[2:]):
        #检查当前维度 s 是否大于等于高斯核的大小
        if s >= win.shape[-1]:
            perms = list(range(win.ndim))#win.ndim表示维度数
            perms[2 + i] = perms[-1]
            perms[-1] = 2 + i
            out = conv(out, weight=win.permute(perms), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out#经过高斯模糊处理的张量处理后的张量 out

#计算了结构相似性指数（SSIM）和对比度敏感性（CS）
def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    #确保高斯核的类型与输入图像 X 相同
    win = win.type_as(X)
    #计算图像的均值和均值平方
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    #计算 SSIM 和 CS
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    #计算每个通道的 SSIM 和 CS 平均值
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs

#用于计算两个图像之间的结构相似性指数（SSIM）
def ssim(X,
         Y,
         data_range=255,#图像数据的动态范围（默认为255）
         size_average=True,#是否对计算结果进行平均（默认为True）。如果为True，返回的是所有像素点的SSIM均值；如果为False，返回每个通道的均值。
         win_size=11,#用于计算SSIM的窗口大小（默认为11）
         win_sigma=1.5,#用于生成高斯窗口的标准差（默认为1.5）
         win=None,#自定义的窗口，如果为None，函数会生成默认的高斯窗口
         K=(0.01, 0.03),
         #是否强制SSIM值非负（默认为False）
         nonnegative_ssim=False):
    # 输出的是灰度图像，其shape是[H, W]
    # 需要扩展为 [B, C, H, W]
    X = TF.to_tensor(X).unsqueeze(0).unsqueeze(0) * 255.0#转换为 PyTorch 张量 形状为 [B, C, H, W] 图像的范围转换为 [0, 255]
    Y = TF.to_tensor(Y).unsqueeze(0).unsqueeze(0) * 255.0
    #确保两张图像的尺寸相同
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    #去除多余维度
    for d in range(len(X.shape) - 1, 1, -1):
        X = torch.squeeze(X, dim=d)#用于去除张量中指定维度上大小为1的维度
        Y = torch.squeeze(Y, dim=d)

    #检查 X 的维度数量是否为4或5
    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #确保两个输入图像 X 和 Y 的数据类型相同
    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")
    #设置窗口大小
    if win is not None:  # set win_size
        win_size = win.shape[-1]
    #窗口大小必须是奇数
    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")
    #没有提供窗口 win 的情况下，生成一个默认的高斯窗口
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)#一个形状为 [1, 1, size] 的一维高斯卷积核
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))#win [C, 1, size] 确保高斯卷积核可以与输入图像的每个通道一致地进行卷积操作

    ssim_per_channel, _ = _ssim(X, Y, data_range=data_range, win=win, K=K)
    if nonnegative_ssim:
        ssim_per_channel = F.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()#返回一个标量值。这个标量表示所有通道的平均 SSIM 值。
    else:
        return ssim_per_channel.mean(dim=1)#张量在通道维度（通常是第二维，即 dim=1）上的均值，并返回一个张量。

#多尺度结构相似性指数（MS-SSIM）计算，通常用于评估两张图像的整体视觉相似度
def ms_ssim(
        X,
        Y,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        win=None,
        weights=None,
        K=(0.01, 0.03)
):
    # 输出的是灰度图像，其shape是[H, W]
    # 需要扩展为 [B, C, H, W]
    X = TF.to_tensor(X).unsqueeze(0).unsqueeze(0) * 255.0
    Y = TF.to_tensor(Y).unsqueeze(0).unsqueeze(0) * 255.0
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])#找到图像的最小边长
    #验证输入图像的尺寸是否足够大，以支持多尺度结构相似性指数
    assert smaller_side > (win_size - 1) * (
            2 ** 4#计算了下采样（池化操作）后图像边缘需要保持的最小尺寸
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.tensor(weights, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []#存储每个尺度的对比度敏感性CS
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, K=K)

        if i < levels - 1:
            mcs.append(F.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = F.relu(ssim_per_channel)#对 SSIM 值应用 ReLU 激活函数 # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.reshape((-1, 1, 1)), dim=0)

    if size_average:
        return ms_ssim_val.mean()#返回一个标量，表示整个批次的平均 MS-SSIM
    else:
        return ms_ssim_val.mean(dim=1)#返回一个张量，表示每个通道的 MS-SSIM 平均值。

#封装在一个 nn.Module 中，方便在 PyTorch 模型中使用
class SSIM(nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
    ):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).tile([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        ).item()

# ms_ssim的封装
class MS_SSIM(nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            weights=None,
            K=(0.01, 0.03),
    ):
        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).tile([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        ).item()

#使用 Sobel 算子计算输入图像 x 的水平和垂直方向的梯度图
def sobel_fn(x):
    # Sobel operators
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8#垂直边缘的 Sobel 卷积核
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8#水平边缘的 Sobel 卷积核

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)
    p, q = x_ext.shape
    gv = np.zeros((p - 2, q - 2))
    gh = np.zeros((p - 2, q - 2))
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')
    # for ii in range(1, p - 1):
    #     for jj in range(1, q - 1):
    #         gv[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * vtemp)
    #         gh[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * htemp)

    return gv, gh#gv 和 gh，即图像的垂直和水平梯度

#对输入图像进行周期性扩展
def per_extn_im_fn(x, wsize):
    """
    Periodic extension of the given image in 4 directions.

    xout_ext = per_extn_im_fn(x, wsize)

    Periodic extension by (wsize-1)/2 on all 4 sides.
    wsize should be odd.

    Example:
        Y = per_extn_im_fn(X, 5);    % Periodically extends 2 rows and 2 columns in all sides.
    """
    # x: 输入的二维图像矩阵。
    # wsize: 扩展的窗口大小，必须为奇数

    hwsize = (wsize - 1) // 2  # Half window size excluding centre pixel.

    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # Row-wise periodic extension.
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    # Column-wise periodic extension.
    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext

#计算了图像融合的性能指标 NABF 测量了融合图像与原始图像之间的相对差异
def get_Nabf(I1, I2, f):
    # Parameters for Petrovic Metrics Computation.
    Td = 2
    wt_min = 0.001
    P = 1
    Lg = 1.5
    Nrg = 0.9999
    kg = 19
    sigmag = 0.5
    Nra = 0.9995
    ka = 22
    sigmaa = 0.5

    xrcw = f.astype(np.float64)
    x1 = I1.astype(np.float64)
    x2 = I2.astype(np.float64)

    # Edge Strength & Orientation.
    gvA, ghA = sobel_fn(x1)
    gA = np.sqrt(ghA ** 2 + gvA ** 2)

    gvB, ghB = sobel_fn(x2)
    gB = np.sqrt(ghB ** 2 + gvB ** 2)

    gvF, ghF = sobel_fn(xrcw)
    gF = np.sqrt(ghF ** 2 + gvF ** 2)

    # Relative Edge Strength & Orientation.
    gAF = np.zeros(gA.shape)
    gBF = np.zeros(gB.shape)
    aA = np.zeros(ghA.shape)
    aB = np.zeros(ghB.shape)
    aF = np.zeros(ghF.shape)
    p, q = xrcw.shape
    maskAF1 = (gA == 0) | (gF == 0)
    maskAF2 = (gA > gF)
    gAF[~maskAF1] = np.where(maskAF2, gF / gA, gA / gF)[~maskAF1]
    maskBF1 = (gB == 0) | (gF == 0)
    maskBF2 = (gB > gF)
    gBF[~maskBF1] = np.where(maskBF2, gF / gB, gB / gF)[~maskBF1]
    aA = np.where((gvA == 0) & (ghA == 0), 0, np.arctan(gvA / ghA))
    aB = np.where((gvB == 0) & (ghB == 0), 0, np.arctan(gvB / ghB))
    aF = np.where((gvF == 0) & (ghF == 0), 0, np.arctan(gvF / ghF))

    aAF = np.abs(np.abs(aA - aF) - np.pi / 2) * 2 / np.pi
    aBF = np.abs(np.abs(aB - aF) - np.pi / 2) * 2 / np.pi

    QgAF = Nrg / (1 + np.exp(-kg * (gAF - sigmag)))
    QaAF = Nra / (1 + np.exp(-ka * (aAF - sigmaa)))
    QAF = np.sqrt(QgAF * QaAF)
    QgBF = Nrg / (1 + np.exp(-kg * (gBF - sigmag)))
    QaBF = Nra / (1 + np.exp(-ka * (aBF - sigmaa)))
    QBF = np.sqrt(QgBF * QaBF)

    wtA = wt_min * np.ones((p, q))
    wtB = wt_min * np.ones((p, q))
    cA = np.ones((p, q))
    cB = np.ones((p, q))
    wtA = np.where(gA >= Td, cA * gA ** Lg, 0)
    wtB = np.where(gB >= Td, cB * gB ** Lg, 0)

    wt_sum = np.sum(wtA + wtB)
    QAF_wtsum = np.sum(QAF * wtA) / wt_sum  # Information Contributions of A.
    QBF_wtsum = np.sum(QBF * wtB) / wt_sum  # Information Contributions of B.
    QABF = QAF_wtsum + QBF_wtsum  # QABF=sum(sum(QAF.*wtA+QBF.*wtB))/wt_sum -> Total Fusion Performance.

    Qdelta = np.abs(QAF - QBF)
    QCinfo = (QAF + QBF - Qdelta) / 2
    QdeltaAF = QAF - QCinfo
    QdeltaBF = QBF - QCinfo
    QdeltaAF_wtsum = np.sum(QdeltaAF * wtA) / wt_sum
    QdeltaBF_wtsum = np.sum(QdeltaBF * wtB) / wt_sum
    QdeltaABF = QdeltaAF_wtsum + QdeltaBF_wtsum  # Total Fusion Gain.
    QCinfo_wtsum = np.sum(QCinfo * (wtA + wtB)) / wt_sum
    QABF11 = QdeltaABF + QCinfo_wtsum  # Total Fusion Performance.

    rr = np.zeros((p, q))
    rr = np.where(gF <= np.minimum(gA, gB), 1, 0)

    LABF = np.sum(rr * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

    na1 = np.where((gF > gA) & (gF > gB), 2 - QAF - QBF, 0)
    NABF1 = np.sum(na1 * (wtA + wtB)) / wt_sum

    # Fusion Artifacts (NABF) changed by B. K. Shreyamsha Kumar.

    na = np.where((gF > gA) & (gF > gB), 1, 0)
    NABF = np.sum(na * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum
    return NABF#评估图像融合质量的指标。它综合考虑了边缘强度、边缘方向以及融合图像与源图像之间的差异


def sobel_fn(x):
    # Sobel operators
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)
    p, q = x_ext.shape
    gv = np.zeros((p - 2, q - 2))
    gh = np.zeros((p - 2, q - 2))
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')
    # for ii in range(1, p - 1):
    #     for jj in range(1, q - 1):
    #         gv[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * vtemp)
    #         gh[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * htemp)

    return gv, gh


def per_extn_im_fn(x, wsize):
    """
    Periodic extension of the given image in 4 directions.

    xout_ext = per_extn_im_fn(x, wsize)

    Periodic extension by (wsize-1)/2 on all 4 sides.
    wsize should be odd.

    Example:
        Y = per_extn_im_fn(X, 5);    % Periodically extends 2 rows and 2 columns in all sides.
    """

    hwsize = (wsize - 1) // 2  # Half window size excluding centre pixel.

    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # Row-wise periodic extension.
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    # Column-wise periodic extension.
    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext

#计算三幅图像（pA, pB, pF）之间的质量评估分数
def get_Qabf(pA, pB, pF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5;
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel Operator Sobel算子
    # h1, h2, h3 分别用于计算图像在不同方向上的梯度响应。
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)


    # if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
    # 如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

    strA = pA
    strB = pB
    strF = pF

    # 数组旋转180度
    def flip180(arr):
        return np.flip(arr)

    # 相当于matlab的Conv2
    # 先对内核进行一次180度翻转，再使用convolve2d，而convolve2d内部再翻转一次内核，所以这个操作的结果等同于直接使用未翻转的Sobel内核进行卷积。
    def convolution(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        img_new = convolve2d(data, k, mode='valid')
        return img_new

    #通过图像的梯度强度和方向计算图像之间的相似性指标
    #gA（梯度强度） aA（梯度方向）
    def getArray(img):
        # 对图像 img 使用 Sobel 算子进行卷积，分别计算 x 方向和 y 方向的梯度。h3 和 h1 是 Sobel 核。
        SAx = convolution(h3, img)
        SAy = convolution(h1, img)

        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))#梯度强度
        n, m = img.shape #高 宽
        aA = np.zeros((n, m))#大小为 (n, m)0数组
        zero_mask = SAx == 0#zero_mask 是一个布尔型数组，SAx中元素 == 0则返回True,反之False 即zero_mask数组中True表示0,False表示非0
        aA[~zero_mask] = np.arctan(SAy[~zero_mask] / SAx[~zero_mask])
        #~zero_mask取反，True表示非0,False表示0 ，SAy[~zero_mask]表示以~zero_mask为索引提取出不为0的元素 aA[~zero_mask] 存储了图像中每个位置的梯度方向
        aA[zero_mask] = np.pi / 2#SAx 为零时的梯度方向被固定为 π / 2
        # for i in range(n):
        #     for j in range(m):
        #         if (SAx[i, j] == 0):
        #             aA[i, j] = math.pi / 2
        #         else:
        #             aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
        return gA, aA

    # 对strB和strF进行相同的操作
    gA, aA = getArray(strA)
    gB, aB = getArray(strB)
    gF, aF = getArray(strF)

    # the relative strength and orientation value of GAF,GBF and AAF,ABF;
    #图像的质量评价指标
    def getQabf(aA, gA, aF, gF):
        mask = (gA > gF)
        #mask值为true则GAF为gF / gA，否则是np.where(gA == gF, gF, gA / gF),np.where(gA == gF, gF, gA / gF)表示gA == gF则gF，否则gA / gF
        GAF = np.where(mask, gF / gA, np.where(gA == gF, gF, gA / gF))#梯度强度的相对值

        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)#方向的相对值 AAF。方向差异越小，AAF 值越接近 1。

        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))# GAF（梯度强度相对值）的质量分数 QgAF
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))# AAF（梯度方向相对值）的质量分数 QaAF

        QAF = QgAF * QaAF
        return QAF#整体质量分数

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)#图像中所有像素点的梯度强度之和
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))#加权的梯度强度之和
    output = nume / deno#最终的图像质量分数 在0到1之间，越高表示图像质量越接近参考图像
    return output

#用于处理图像数据，并根据指定的特征提取方法来分析图像对
def analysis_fmi(ima, imb, imf, feature, w):
    #输入图像转换为 double 数据类型
    ima = np.double(ima)
    imb = np.double(imb)
    imf = np.double(imf)
    #feature: 指定的特征提取方法 选项包括 'none'（原始像素）、'gradient'（梯度）、'edge'（边缘）、'dct'（DCT）、'wavelet'（小波变换）
    # w: 用于特征提取的窗口大小
    # Feature Extraction
    if feature == 'none':  # Raw pixels (no feature extraction)
        aFeature = ima
        bFeature = imb
        fFeature = imf
    elif feature == 'gradient':  # Gradient
        aFeature = generic_gradient_magnitude(ima, sobel)
        bFeature = generic_gradient_magnitude(imb, sobel)
        fFeature = generic_gradient_magnitude(imf, sobel)
    elif feature == 'edge':  # Edge
        aFeature = np.double(sobel(ima) > w)
        bFeature = np.double(sobel(imb) > w)
        fFeature = np.double(sobel(imf) > w)
    elif feature == 'dct':  # DCT
        aFeature = dctn(ima, type=2, norm='ortho')
        bFeature = dctn(imb, type=2, norm='ortho')
        fFeature = dctn(imf, type=2, norm='ortho')
    elif feature == 'wavelet':  # Discrete Meyer wavelet
        raise NotImplementedError('Wavelet feature extraction not yet implemented in Python!')
    else:
        raise ValueError(
            "Please specify a feature extraction method among 'gradient', 'edge', 'dct', 'wavelet', or 'none' (raw "
            "pixels)!")

    m, n = aFeature.shape
    w = w // 2
    fmi_map = np.ones((m - 2 * w, n - 2 * w))

#对一组图像进行评估，并计算各种指标的均值。
'''
eval_flag:指示是否进行正式评估。如果 eval_flag 为 None（通常意味着测试模式），则函数会仅选择前 20 张图像进行评估。如果设置为 True（表示正式评估），则函数会使用所有可用的图像进行评估。
dataroot:用途: 数据集的根目录路径。包含图像数据的路径。如果未提供，函数会使用默认路径 
results_root: 存放结果图像的根目录路径。如果未提供，函数会使用默认路径。
dataset: 指定要使用的数据集名称。函数会在 dataroot 和 results_root 路径下查找该数据集的文件夹。默认为 'TNO'。
easy_flag: 是否启用简单模式或容易模式。在 evaluation_one 函数中，可能会根据 easy_flag 的值选择不同的评估方法或参数设置。默认为 False。
默认值: False
'''
# def eval_multi_method(eval_flag=None, dataroot=None, results_root=None, dataset=None, easy_flag=False):
#     if dataroot is None:
#         dataroot = r'D:\project\evalueation\dataset'
#     if results_root is None:
#         results_root = r'D:\project\evalueation\result'
#     if dataset is None:
#         dataset = 'TNO'
#     ir_dir = os.path.join(dataroot, dataset, 'Inf')
#     vi_dir = os.path.join(dataroot, dataset, 'Vis')
#     f_dir = os.path.join(results_root, dataset)
#     filelist = natsorted(os.listdir(ir_dir))# 获取红外图像目录中的文件列表，并按自然排序
#     if eval_flag is False:   # 如果是正式评估，需要 eval_flag=True
#         file_list = []
#         for i in range(20):     # 42张图太久了，只要前20张
#             file_list.append(filelist[i])
#     else:
#         file_list = filelist
#
#     # 初始化各种评估指标的列表
#     EN_list = []
#     MI_list = []
#     SF_list = []
#     AG_list = []
#     SD_list = []
#     CC_list = []
#     SCD_list = []
#     VIF_list = []
#     MSE_list = []
#     PSNR_list = []
#     Qabf_list = []
#     Nabf_list = []
#     SSIM_list = []
#     MS_SSIM_list = []
#
#
#     sub_f_dir = os.path.join(f_dir)
#     eval_bar = tqdm(file_list)## 创建进度条显示
#     for _, item in enumerate(eval_bar):
#         ir_name = os.path.join(ir_dir, item)# 构建红外图像文件路径
#         vi_name = os.path.join(vi_dir, item)
#         f_name = os.path.join(sub_f_dir, item)
#         # print(ir_name, vi_name, f_name)
#         ## 调用评估函数
#         EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir_name, vi_name,
#                                                                                                 f_name, easy_flag)
#         # 将评估结果添加到相应的列表中
#         EN_list.append(EN)
#         MI_list.append(MI)
#         SF_list.append(SF)
#         AG_list.append(AG)
#         SD_list.append(SD)
#         CC_list.append(CC)
#         SCD_list.append(SCD)
#         VIF_list.append(VIF)
#         MSE_list.append(MSE)
#         PSNR_list.append(PSNR)
#         Qabf_list.append(Qabf)
#         Nabf_list.append(Nabf)
#         SSIM_list.append(SSIM)
#         MS_SSIM_list.append(MS_SSIM)
#
#         # 更新进度条描述
#         eval_bar.set_description("Eval | {}".format(item))
#
#     # 均值
#     EN_mean = np.mean(EN_list)
#     MI_mean = np.mean(MI_list)
#     SF_mean = np.mean(SF_list)
#     AG_mean = np.mean(AG_list)
#     SD_mean = np.mean(SD_list)
#     CC_mean = np.mean(CC_list)
#     SCD_mean = np.mean(SCD_list)
#     VIF_mean = np.mean(VIF_list)
#     MSE_mean = np.mean(MSE_list)
#     PSNR_mean = np.mean(PSNR_list)
#     Qabf_mean = np.mean(Qabf_list)
#     Nabf_mean = np.mean(Nabf_list)
#     SSIM_mean = np.mean(SSIM_list)
#     MS_SSIM_mean = np.mean(MS_SSIM_list)
#     return EN_mean, MI_mean, SF_mean, AG_mean, SD_mean, CC_mean, SCD_mean, VIF_mean, \
#            MSE_mean, PSNR_mean, Qabf_mean, Nabf_mean, SSIM_mean, MS_SSIM_mean


#用于评估不同图像融合算法性能的主程序。会遍历多个数据集和算法，并记录每种算法在不同数据集上的各种质量评估指标的结果

# if __name__ == '__main__':
#     #需要评估的图像融合算法列表
#     algorithms = ['DenseFuse', 'RFN-Nest', 'FusionGAN', 'IFCNN', 'PMGI', 'SDNet',
#                   'U2Fusion', 'FLFuse', 'SeAFusion', 'PIAFusion']
#     #需要使用的数据集列表
#     datasets = ['TNO', 'RoadScene', 'MSRS', 'M3FD']         # 'TNO', 'RoadScene', 'MSRS', 'M3FD'
#     dataroot = r'D:\project\evalueation\dataset'    # 数据集路径
#     results_root = r'D:\project\evalueation\result'    # 算法结果路径
#
#     log_path = r'D:\project\pythonTh_poject\OwnFusion\logs'#日志文件存储路径
#     logger = logging.getLogger()#设置日志记录器，用于记录评估过程中的信息
#     setup_logger(log_path)
#
#     for i in range(len(datasets)):
#         dataset = datasets[i]
#         logger.info('Dataset: {}'.format(dataset))#用于记录信息的日志语句,它的作用是将当前数据集的名称记录到日志中
#         #用于在 Python 中生成格式化表格的库
#         table = PrettyTable(['Algorithm', 'EN', 'MI', 'SF', 'AG', 'SD', 'CC', 'SCD',
#                              'VIF', 'MSE', 'PSNR', 'Qabf', 'Nabf', 'SSIM', 'MS_SSIM'])#表格的列标题
#         for j in range(len(algorithms)):
#             algorithm = algorithms[j]
#             logger.info('Algorithm: {}'.format(algorithm))
#             fused_path = os.path.join(results_root, algorithm)
#             EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = \
#                 eval_multi_method(False, dataroot, fused_path, dataset)
#             #round(EN, 4) 保留小数点后4位
#             table.add_row([str(algorithm), round(EN, 4), round(MI, 4), round(SF, 4), round(AG, 4), round(SD, 4),
#                            round(CC, 4), round(SCD, 4), round(VIF, 4), round(MSE, 4), round(PSNR, 4),
#                            round(Qabf, 4), round(Nabf, 4), round(SSIM, 4), round(MS_SSIM, 4)])
#         logger.info(table.get_string())

import csv

def eval_multi_method(eval_flag=None, dataroot=None, results_root=None, dataset=None, easy_flag=False):
    if dataroot is None:
        dataroot = r'D:\project\evalueation\dataset'
    if results_root is None:
        results_root = r'D:\project\evalueation\result'
    if dataset is None:
        dataset = 'TNO'

    ir_dir = os.path.join(dataroot, dataset, 'Inf')  # 红外图像目录
    vi_dir = os.path.join(dataroot, dataset, 'Vis')  # 可见光图像目录
    f_dir = os.path.join(results_root, dataset)  # 融合图像结果目录
    filelist = natsorted(os.listdir(ir_dir))  # 获取红外图像目录中的文件列表，并按自然排序

    # 如果eval_flag是False，只评估前20张图片
    if eval_flag is False:
        file_list = []
        for i in range(100):  # 可以根据需要调整评估图片的数量
            file_list.append(filelist[i])
    else:
        file_list = filelist

    # 初始化各指标的列表，用来保存每张图片的评估结果
    EN_list = []
    MI_list = []
    SF_list = []
    AG_list = []
    SD_list = []
    CC_list = []
    SCD_list = []
    VIF_list = []
    MSE_list = []
    PSNR_list = []
    Qabf_list = []
    Nabf_list = []
    SSIM_list = []
    MS_SSIM_list = []

    sub_f_dir = os.path.join(f_dir)
    eval_bar = tqdm(file_list)  # 进度条

    for _, item in enumerate(eval_bar):
        ir_name = os.path.join(ir_dir, item)  # 红外图像路径
        vi_name = os.path.join(vi_dir, item)  # 可见光图像路径
        f_name = os.path.join(sub_f_dir, item)  # 融合图像路径

        # 调用evaluation_one函数计算该图片的评估指标
        EN, MI, SF, AG, SD, VIF, SCD = evaluation_one(ir_name, vi_name, f_name, easy_flag)

        # 将每张图片的指标添加到各自的列表中
        EN_list.append(EN)
        MI_list.append(MI)
        SF_list.append(SF)
        AG_list.append(AG)
        SD_list.append(SD)
        # CC_list.append(CC)
        SCD_list.append(SCD)
        VIF_list.append(VIF)
        # MSE_list.append(MSE)
        # PSNR_list.append(PSNR)
        # Qabf_list.append(Qabf)
        # Nabf_list.append(Nabf)
        # SSIM_list.append(SSIM)
        # MS_SSIM_list.append(MS_SSIM)

        # 更新进度条的描述
        eval_bar.set_description(f"Eval | {item}")

    return file_list, EN_list, MI_list, SF_list, AG_list, SD_list, CC_list, SCD_list, VIF_list, \
           MSE_list, PSNR_list, Qabf_list, Nabf_list, SSIM_list, MS_SSIM_list


if __name__ == '__main__':
    # 需要评估的图像融合算法列表
    algorithms = ['OwnFusion','DenseFuse', 'RFN-Nest', 'FusionGAN', 'IFCNN', 'SDNet',
                  'U2Fusion', 'FLFuse', 'SeAFusion', 'PIAFusion']
    # 需要使用的数据集列表
    datasets = ['M3FD']  # 'TNO', 'RoadScene', 'MSRS', 'M3FD'
    dataroot = r'D:\project\evalueation\dataset'  # 数据集路径
    results_root = r'D:\project\evalueation\result'  # 算法结果路径

    log_path = r'D:\project\pythonTh_poject\OwnFusion\logs'  # 日志文件存储路径
    logger = logging.getLogger()  # 设置日志记录器，用于记录评估过程中的信息
    setup_logger(log_path)
    #注意**********************************************************
    # 打开CSV文件以记录每张图片的评估结果
    with open('evaluation_results_M3FD.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV文件的表头
        writer.writerow(['Algorithm', 'Image', 'EN', 'MI', 'SF', 'AG', 'SD',  'VIF',  'SCD'])

        for i in range(len(datasets)):
            dataset = datasets[i]
            logger.info('Dataset: {}'.format(dataset))  # 用于记录信息的日志语句,它的作用是将当前数据集的名称记录到日志中
            for j in range(len(algorithms)):
                algorithm = algorithms[j]
                logger.info('Algorithm: {}'.format(algorithm))
                fused_path = os.path.join(results_root, algorithm)

                # 获取每张图片的评估结果
                file_list, EN_list, MI_list, SF_list, AG_list, SD_list, CC_list, SCD_list, VIF_list, MSE_list, PSNR_list, Qabf_list, Nabf_list, SSIM_list, MS_SSIM_list = \
                    eval_multi_method(False, dataroot, fused_path, dataset)

                # 逐张图片记录到CSV文件中
                # for idx in range(len(file_list)):
                #     writer.writerow([algorithm, file_list[idx], round(EN_list[idx], 4), round(MI_list[idx], 4),
                #                      round(SF_list[idx], 4), round(AG_list[idx], 4), round(SD_list[idx], 4),
                #                      round(CC_list[idx], 4), round(SCD_list[idx], 4), round(VIF_list[idx], 4),
                #                      round(MSE_list[idx], 4), round(PSNR_list[idx], 4), round(Qabf_list[idx], 4),
                #                      round(Nabf_list[idx], 4), round(SSIM_list[idx], 4), round(MS_SSIM_list[idx], 4)])
                for idx in range(len(file_list)):
                    writer.writerow([algorithm, file_list[idx], round(EN_list[idx], 4), round(MI_list[idx], 4),
                                     round(SF_list[idx], 4), round(AG_list[idx], 4), round(SD_list[idx], 4),
                                     round(VIF_list[idx], 4),round(SCD_list[idx], 4)])


