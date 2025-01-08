import torch
import torch.nn as nn
from torchsummary import summary  #summary 来展示 PyTorch 模型的结构和参数信息。
from RepRFN import RepBlock

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

# 完整主模型
class easyFusion(nn.Module):
    def __init__(self, deploy=False):
        super(easyFusion, self).__init__()
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

        self.conv1 = RepBlock(ch[4], ch[3], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2 = RepBlock(ch[3], ch[2], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv3 = RepBlock(ch[2], ch[1], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv4 = nn.Conv2d(ch[1], ch[0], kernel_size=1, padding=0)

        self.act = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, image_vi, image_ir):
        # encoder
        x_vi = self.act(self.conv0_vi(image_vi))
        x_ir = self.act(self.conv0_ir(image_ir))

        #第一个RepBlock模块
        x_vi = self.conv1_vi(x_vi)
        x_ir = self.conv1_ir(x_ir)


        #第二个RepBlock模块
        x_vi = self.conv2_vi(x_vi)
        x_ir = self.conv2_ir(x_ir)


        # fusion
        # x = torch.cat((x_vi , x_ir), dim=1)
        x = torch.cat([(x_vi * x_ir), (self.CBAM1(x_vi) + self.CBAM2(x_ir))], dim=1)


        # decoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.tanh(self.conv4(x))
        return x / 2 + 0.5#将输出范围从 [-1, 1] 变换到 [0, 1]

#遍历所有模块
def model_deploy(model):
    for module in model.modules():
        #检查模块是否具有 switch_to_deploy 方法，返回一个布尔值，如果模块 module 中存在 switch_to_deploy 方法，返回 True；否则返回 False。
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    return model


#模型性能测试脚本
def unit_test():
    import time
    n = 100  # 循环次数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 1, 480, 640).cuda()
    model = easyFusion().to(device)
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