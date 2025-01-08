import datetime
import math
import os
import time
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch import nn



#################################################################################
# 图像处理函数：ImageProcessing
# 对融合初图像进行一些处理
#################################################################################
def ImageProcessing(fusion_image):
    fusion_image = torch.clamp(fusion_image, max=1.0, min=0.0)      # 限制在[0,1]，不然会出现耀斑
    # 转回CPU，因为之前可能加载到GPU了
    fused_image = fusion_image.cpu().detach().numpy()#detach 从计算图中分离张量，使其不再包含梯度信息
    fused_image = fused_image.transpose((0, 2, 3, 1))#(batch, height, width, channel)
    # 均衡像素强度
    # fused_image = (fused_image - np.min(fused_image)) / (np.max(fused_image) - np.min(fused_image))
    fused_image = np.uint8(255.0 * fused_image)#转换为常见的图像文件格式所需要的格式，即 8 位无符号整数。
    return fused_image


#################################################################################
# 运行时间处理函数：RunningTime
# 对训练的运行时间进行一些处理
#################################################################################
class RunningTime:
    def __init__(self):
        self.start_time = time.time()#每个训练循环开始的时间
        self.init_time = time.time()#整个训练过程开始的时间
        self.end_time = 0#每个训练循环结束的时间
        self.this_time = 0#每个训练循环的持续时间
        self.total_time = 0#从训练开始到当前时间的总持续时间
        self.now_it = 0#记录当前模型训练过的样本数量
        self.eta = 0#估计剩余时间（Estimated Time of Arrival）

    def runtime(self, this_epo, it, dataset_size, epoch,batch_size):
        #dataset_size可以理解为一轮的批次数
        self.end_time = time.time()
        self.this_time = self.end_time - self.start_time
        self.total_time = self.end_time - self.init_time
        #this_epo:当前的 epoch 数  it: 当前 epoch 内的第几批
        # self.now_it = dataset_size * this_epo + it + 1
        self.now_it = dataset_size * this_epo + (it + 1) * batch_size
        #剩余样本数量*单个样本训练时间=估计剩余时间
        self.eta = int((dataset_size * epoch - self.now_it) * (self.total_time / self.now_it))
        #以更易读的格式（如“小时:分钟:秒”）显示
        self.eta = str(datetime.timedelta(seconds=self.eta))
        self.start_time = self.end_time
        return self.eta, self.this_time, self.now_it


#################################################################################
# 单个特征可视化函数
# 在模型的forward阶段返回一个特征值，然后将其保存为各通道一起的灰度图像
#################################################################################
# 用于保存特征图（feature map）到指定路径的图像文件中
# fmap: 输入的特征图张量
# save_path: 保存图像的路径
# fmap_size: 特征图要调整到的尺寸大小
# i: 一个标识符，用于命名保存的图像文件
def feature_save(fmap, save_path=None, fmap_size=None, i=None):
    if fmap_size is None:
        _, _, H, W = fmap.shape
        fmap_size = [H, W]
    fmap = torch.unsqueeze(fmap[-1], 0)  # 选取最后一个样本，并将其形状从 (C, H, W) 调整为 (1, C, H, W),保持原来的尺寸，只要一张图的tensor
    fmap.transpose_(0, 1)  # 等价于 x = x.transpose(0,1), 把B和C的维度调换 从 (1, C, H, W) 变为 (C, 1, H, W)
    nrow = int(np.sqrt(fmap.shape[0]))#计算通道数的平方根，确定每行应该放置多少个图像，以生成接近正方形的图像网格。
    fmap = F.interpolate(fmap, size=fmap_size, mode="bilinear")  # 改变数组尺寸，变成[64, 64]
    fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)#将特征图整理成一个网格图像
    if not os.path.exists(save_path):     # 如果没有路径文件夹，就创建文件夹
        os.makedirs(save_path)
        print('Making fused_path {}'.format(save_path))
    save_path = os.path.join(save_path, 'feature{}.png'.format(i))
    vutils.save_image(fmap_grid, save_path)#将特征图网格保存为图像文件，并打印保存路径以确认
    print('Save feature map {}'.format(save_path))


#################################################################################
# 构建K=7，sigmaS=25，sigmac=7.5的双边滤波
#################################################################################
# 用于生成一个高斯卷积核
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    center = ksize // 2 #计算卷积核的中心索引
    xs = (np.arange(ksize, dtype=np.float32) - center)  # 生成一个一维数组，表示每个元素与矩阵中心的距离[-3. -2. -1.  0.  1.  2.  3.]
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))  # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()  # 归一化
    return kernel

'''

函数 bilateralFilter，用于实现双边滤波（Bilateral Filtering），一种用于图像去噪和边缘保持的滤波技术。
双边滤波结合了空间域和颜色域的权重，能够在去除噪声的同时保留图像的边缘信息。
batch_img: 输入的图像批次，形状为 (B, C, H, W)
sigmaColor: 颜色域的标准差，用于控制颜色相似性对滤波权重的影响,默认为 0。
sigmaSpace: 空间域的标准差，用于控制空间距离对滤波权重的影响,默认为 0。

'''
def bilateralFilter(batch_img, ksize, sigmaColor=0., sigmaSpace=0.):
    device = batch_img.device
    #初始化 sigmaSpace 和 sigmaColor
    if sigmaSpace == 0:#空间距离
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor == 0:#像素颜色差
        sigmaColor = sigmaSpace

    #图像边缘填充
    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    # batch_img 的维度为 BxcxHxW, 经过边缘填充后的批量图像，形状为 (B, C, H_pad, W_pad)，因此要沿着第 二、三维度即H_pad、W_pad进行 unfold
    # patches.shape: B x C x H x W x ksize x ksize
    # unfold 是 PyTorch 中的一个方法，用于从张量的某个维度提取滑动窗口Tensor.unfold(dimension, size, step)
    # 最终的补丁张量 patches 的形状为 (B, C, H_new, W_new, ksize, ksize) H_new 表示滑动窗口在高度方向上可以提取到的补丁数量
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6

    # 求出像素亮度差 结果是一个形状为 (B, C, H_new, W_new, ksize, ksize) 的张量，表示每个补丁内的每个像素的差异值。
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # 根据像素亮度差，计算权重矩阵 weights_color 是一个张量，表示每个像素在颜色空间中的权重。形状为 (B, C, H_new, W_new, ksize, ksize)
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)
    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)#使得 weights_space 和 weights_color 的形状匹配
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # 两个权重矩阵相乘得到总的权重矩阵 (B, C, H_new, W_new, ksize, ksize)
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))#(B, C, H_new, W_new)
    # 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix


# -----------------------------------------------#
# 构建K=7，主要调用这个
# -----------------------------------------------#
# 将双边滤波器集成到 PyTorch 神经网络中
class bilateral_Filter(nn.Module):
    def __init__(self, ksize=5, sigmacolor=0., sigmaspace=0.):
        super(bilateral_Filter, self).__init__()
        self.ksize = ksize
        self.sigmacolor = sigmacolor
        self.sigmaspace = sigmaspace

    def forward(self, x):
        x = bilateralFilter(x, self.ksize, self.sigmacolor, self.sigmaspace)
        return x


#################################################################################
# 色域转换
#   底层函数：RGB2YCrCb、YCrCb2RGB
#   顶层函数：ColorSpaceTransform
#################################################################################
# 底层函数，色域转换基本函数
def RGB2YCrCb(input_im):
    #举例：原始形状 [2, 3, 4, 5] 转换为展平后的形状 [40, 3]
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    #R、G 和 B 分别提取了所有图像中每个像素的 R、G 和 B 值
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    # YCrCb 颜色空间的转换公式计算亮度通道 Y 的值
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    #维度扩展 由[N]变成[N, 1]
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    #拼接得到一个形状为 [N, 3] 的张量，每行包含一个像素的三个通道值：[Y, Cr, Cb]
    temp = torch.cat((Y, Cr, Cb), dim=1)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )#temp 重塑为 [B, H, W, 3]
            .transpose(1, 3)
            .transpose(2, 3)#[B, C, H, W]
    )
    return out

#将 YCrCb 颜色空间的图像转换回 RGB 颜色空间。
def YCrCb2RGB(input_im):
    #im_flat 是一个形状为 [N, 3] 的张量，其中 N 是所有像素的总数，3 是 Y、Cr 和 Cb 通道。
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    # mat 是一个转换矩阵，用于将 YCrCb 颜色空间转换回 RGB 颜色空间。
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(input_im.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(input_im.device)
    #temp 是转换后的 RGB 数据，形状仍然是 [N, 3]
    temp = (im_flat + bias).mm(mat).to(input_im.device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    #形状为 [B, 3, H, W] 的张量，表示转换回 RGB 颜色空间的图像
    return out


# 顶层函数，model选择转换模式，images_input输入图像
def ColorSpaceTransform(model, images_input):
    if model == 'RGB2YCrCb':
        images_vis_ycrcb = RGB2YCrCb(images_input)
        return images_vis_ycrcb
    elif model == 'YCrCb2RGB':
        fusion_image = YCrCb2RGB(images_input)
        return fusion_image

#计算并输出给定运行时间的平均值和标准差
def algorithm_runtime(runtime):
    t_avg = np.mean(runtime)
    t_std = np.std(runtime)
    print('t_avg is {:5f}±{:5f}s'.format(t_avg, t_std))

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

if __name__ == '__main__':
    img_path = r"D:\project\pythonTh_poject\OwnFusion\datasets\MSRS\test\Vis\00055D.png"
    img = Image.open(img_path).convert('L')  # 将图像转换为灰度
    # 创建转换器
    transform = transforms.ToTensor()
    img_tensor=transform(img).unsqueeze(0).cuda()
    net=Sobelxy().cuda()
    img_sobel=net(img_tensor)
    feature_save(img_sobel, save_path='./feature_show')
