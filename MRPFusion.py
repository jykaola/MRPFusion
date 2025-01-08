import torch
import torch.nn as nn
from torchsummary import summary  #summary 来展示 PyTorch 模型的结构和参数信息。
from RepRFN import RepBlock
from einops import rearrange
import torch.nn.functional as F

#CBAM层的实现，它被设计成一个PyTorch模块，可以方便地插入到其他神经网络结构
class CBAMLayer(nn.Module):
    #channel输入特征图的通道数
    #reduction 用于定义在通道注意力模块中降低通道数的比例。默认值为16，
    #意味着如果输入特征图有1024个通道，那么中间层将会有 1024 / 16 = 64 个通道。这有助于减少计算量和模型复杂度
    #spatial_kernel 空间注意力模块中卷积核的大小。默认值为7
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        # 定义了一个最大池化层将输入的特征图压缩成一个固定大小的输出，这里是1x1的大小。无论输入特征图尺寸如何，输出都是通道数相同但高宽为1的特征图
        # 用于从每个特征图的通道中提取最显著的特征响应。
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #同样将输入特征图压缩成1x1的输出，但使用的是平均值而非最大值。这有助于捕捉整个特征图的全局平均信息。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP多层感知机的实现
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            # 这里1表示卷积核大小，卷积核为1，则只改变通道数，不改变尺寸。
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        x = self.sigmoid(max_out + avg_out) * x

        #max_out表示从x中每一个像素点选择最大值，这个像素点是每个通道的这个像素点的位置进行选取，然后形成一个张量[N, C, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1))) * x
        return x
#CAFM实现
class Attention(nn.Module):
    #dim: 输入特征的通道数 num_heads: 注意力头的数量 temperature 是一个可学习的参数，用于调节注意力分数的温度
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        #将输入特征图映射为查询（Q）、键（K）和值（V）的通道，输出通道数为 dim * 3
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        #进行深度卷积，用于增强特征提取能力
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        #将最后的输出特征映射回原始的通道数
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        #用于整合多个头的输出，输出通道数为 9
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True, groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        #扩展了一个深度的维度，使得输入形状变为 (b, c, 1, h, w) 批量大小为 b，通道数为 c，深度为 1，高度为 h，宽度为 w
        x = x.unsqueeze(2)
        #self.qkv(x)将输入 x 转换为一个新的特征图，其输出形状为 (b, dim * 3, 1, h, w) 接着，通过 self.qkv_dwconv(...) 使用深度卷积来增强特征提取能力
        #输出的 qkv 形状仍然是 (b, dim * 3, 1, h, w)
        qkv = self.qkv_dwconv(self.qkv(x))
        #去除多余维度: squeeze(2) 将维度 2 移除，恢复为 (b, dim * 3, h, w)
        qkv = qkv.squeeze(2)
        #f_conv 的形状变为 (b, h, w, dim * 3)
        f_conv = qkv.permute(0, 2, 3, 1)
        #reshape(...)：将重塑为形状 (b, h * w, 3 * self.num_heads, -1) 3 * self.num_heads 表示将 Q、K、V 特征通道整合在一起
        #(b, 3 * self.num_heads, h * w, dim // self.num_heads)，其中 dim // self.num_heads 是每个头处理的通道数。
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        #计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        #加权求和值
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output

class CMDAF(nn.Module):
    def __init__(self):
        super(CMDAF, self).__init__()
        # 初始化 alpha 和 beta 为可学习参数
        self.alpha = nn.Parameter(torch.ones(1))  # 初始值为 1，表示默认权重
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, F_vi, F_ir):
        # 计算可见光图像和红外图像之间的逐元素差值
        sub_vi_ir = F_vi - F_ir
        sub_w_vi_ir = torch.mean(sub_vi_ir, dim=[2, 3], keepdim=True)  # 全局平均池化
        w_vi_ir = torch.sigmoid(sub_w_vi_ir)  # 通道权重

        # 计算红外图像和可见光图像之间的逐元素差值
        sub_ir_vi = F_ir - F_vi
        sub_w_ir_vi = torch.mean(sub_ir_vi, dim=[2, 3], keepdim=True)  # 全局平均池化
        w_ir_vi = torch.sigmoid(sub_w_ir_vi)  # 通道权重

        # 放大红外图像对可见光图像的补偿信号
        F_dvi = w_vi_ir * sub_ir_vi  # 放大红外对可见光的补偿
        # 放大可见光图像对红外图像的补偿信号
        F_dir = w_ir_vi * sub_vi_ir

        # 融合时，保留部分原始特征并加入补偿信号
        F_fvi = F_vi + F_dir + self.alpha * F_vi  # 增加 alpha 参数来平衡保留的原始特征
        F_fir = F_ir + F_dvi + self.beta * F_ir  # 增加 beta 参数

        return F_fvi, F_fir

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上

class AveragePoolingLayer(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AveragePoolingLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size  # 如果没有指定步幅，则使用内核大小
        self.padding = padding

    def forward(self, x):
        # 使用平均池化
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

# 完整主模型
class MRPFusion(nn.Module):
    def __init__(self, deploy=False):
        super(MRPFusion, self).__init__()
        ch = [1, 16, 32, 64, 128]
        self.act_type = 'lrelu'
        self.conv0_vi = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)
        self.conv0_ir = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)
        self.conv1_vi = RepBlock(ch[1], ch[2], 2, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv1_ir = RepBlock(ch[1], ch[2], 2, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2_vi = RepBlock(ch[2], ch[3], 2, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2_ir = RepBlock(ch[2], ch[3], 2, act_type=self.act_type, with_idt=False, deploy=deploy)

        self.CBAM1 = CBAMLayer(ch[3])
        self.CBAM2 = CBAMLayer(ch[3])
        # self.SE1 = SE_Block(ch[3])
        # self.SE2 = SE_Block(ch[3])

        # self.conv1x1=nn.Conv2d(ch[3],ch[3],1)

        # 通道调整卷积层，用于跳跃连接
        self.adjust_skip2 = nn.Conv2d(ch[2], ch[4], kernel_size=1, padding=0)
        self.adjust_skip1 = nn.Conv2d(ch[1], ch[3], kernel_size=1, padding=0)
        # self.adjust = nn.Conv2d(64, 128, kernel_size=1, padding=0)


        # 解码器层
        self.conv1 = RepBlock(ch[4], ch[3], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2 = RepBlock(ch[3], ch[2], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv3 = RepBlock(ch[2], ch[1], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv4 = nn.Conv2d(ch[1], ch[0], kernel_size=1, padding=0)

        self.act = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, image_vi, image_ir):
        # 编码器部分
        x_vi = self.act(self.conv0_vi(image_vi))
        x_ir = self.act(self.conv0_ir(image_ir))
        skip1_vi = x_vi  # 保存第一个跳跃连接
        skip1_ir = x_ir

        # 第一个RepBlock模块
        x_vi = self.conv1_vi(x_vi)
        x_ir = self.conv1_ir(x_ir)
        skip2_vi = x_vi  # 保存第二个跳跃连接
        skip2_ir = x_ir

        # 第二个RepBlock模块
        x_vi = self.conv2_vi(x_vi)#64通道
        x_ir = self.conv2_ir(x_ir)
        # x = torch.cat([(x_vi * x_ir), (self.CBAM1(x_vi) + self.CBAM2(x_ir))], dim=1)
        x = torch.cat([(self.CBAM1(x_vi) * self.CBAM1(x_ir)), (self.CBAM1(x_vi) + self.CBAM2(x_ir))], dim=1)
        # 解码器部分（带跳跃连接）
        x = self.conv1(x + self.adjust_skip2(skip2_vi + skip2_ir))  # 调整skip2通道并加到第一个解码器层
        x = self.conv2(x + self.adjust_skip1(skip1_vi + skip1_ir))  # 调整skip1通道并加到第二个解码器层
        x = self.conv3(x)
        x = self.tanh(self.conv4(x))
        return x / 2 + 0.5  # 输出范围从 [-1, 1] 变换到 [0, 1]




# 遍历所有模块
def model_deploy(model):
    for module in model.modules():
        #检查模块是否具有 switch_to_deploy 方法，返回一个布尔值，如果模块 module 中存在 switch_to_deploy 方法，返回 True；否则返回 False。
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    return model


#模型性能测试脚本
def unit_test():
    import time
    n = 20  # 循环次数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 1, 480, 640).cuda()
    model = MRPFusion().to(device)
    model.eval()
    #第一次运行以消除初始化开销
    for i in range(10):
        train_y = model(x, x)    # 第一次运行的不是准确的时间，可能还要加载模型之类的操作

    start_time = time.time()
    for i in range(n):
        train_y = model(x, x)
    train_y_time = time.time() - start_time
    #打印模型的结构总结 显示模型的层次结构、每层的参数量以及输出形状等信息。
    print(summary(model, [(1, 480, 640), (1, 480, 640)]))
    model = model_deploy(model)
    print(summary(model, [(1, 480, 640), (1, 480, 640)]))

    for i in range(10):
        train_y = model(x, x)  # 第一次运行的不是准确的时间，可能还要加载模型之类的操作

    start_time = time.time()
    for i in range(n):
        deploy_y = model(x, x)
    deploy_y_time = time.time() - start_time



    print('train__y time is {:.4f}s/it'.format(train_y_time / n))
    print('deploy_y time is {:.4f}s/it'.format(deploy_y_time / n))
    print('The different is', (train_y - deploy_y).sum())




if __name__ == '__main__':
    unit_test()