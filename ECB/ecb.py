import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqConv3x3(nn.Module):
    #seq_type指定序列卷积的类型
    '''
    SeqConv3x3 类支持四种类型的序列卷积：
    'conv1x1-conv3x3': 先执行一个1x1卷积，然后是一个3x3卷积。
    'conv1x1-sobelx': 先执行一个1x1卷积，然后应用Sobel水平边缘检测算子。
    'conv1x1-sobely': 先执行一个1x1卷积，然后应用Sobel垂直边缘检测算子。
    'conv1x1-laplacian': 先执行一个1x1卷积，然后应用Laplacian边缘检测算子。

    inp_planes 表示输入特征图的通道数
    out_planes 输出特征图的通道数
    depth_multiplier 调整第一个1x1卷积层的输出通道数，1x1卷积通常用于降维或升维，即减少或增加特征图的通道数
    depth_multiplier可以大于1、等于1或小于1，分别用于增加、保持不变或减少输出通道数
    '''
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            #定位于两个卷积操作（1x1卷积和3x3卷积）之间的特征图的通道数
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        #用于应用Sobel边缘检测算子中的水平方向边缘检测
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias


            '''
            Sobel算子不改变通道数，所以在Sobel之前的1*1卷积操作，直接从inp_planes到out_planes
            初始化Sobel算子的参数，包括尺度（scale）、偏置（bias）和掩码（mask）
            scale和bias都是可学习的参数，它们用于缩放和偏移Sobel算子的结果
            self.out_planes表示数道数相匹配，后三维度1 x 1 x 1会通过张量“广播”到与输出特征图相同的形状
            其中的元素是从均值为0、标准差为1的正态分布中随机采样的。这意味着每个通道的scale值都是随机的，而且可能正也可能负。
            随后的 * 1e-3操作则是将这些随机生成的值全部乘以1e - 3（即0.001）
            '''

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            #生成一个随机分布的张量将其乘以一个小的常数 1e-3
            #scale被包装成nn.Parameter后，被添加到模块（model）的参数列表中，反向传播过程中，scale的值就会根据损失函数的梯度自动更新
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            #重塑为一个形状为(self.out_planes,)的一维张量
            bias = torch.reshape(bias, (self.out_planes,))
            #torch.reshape()：这是 PyTorch 中用于改变张量形状的方法。它接受两个参数：第一个参数是要被重塑的张量。
            # 第二个参数是一个元组，指定了新的形状。这里的形状是一个只包含一个元素的元组，即 (self.out_planes,)。
            self.bias = nn.Parameter(bias)
            # 创建了一个形状为(self.out_planes, 1, 3, 3)的张量, self.out_planes 是输出特征图的通道数,
            # 1 表示输入特征图的通道输。需要注意：mask并不是一个卷积核，可以把mask看作out_planes深度为1的卷积核，也就是说
            # 一个深度为1的卷积核作用于一个输入特征图的通道
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)#Sobel算子中，掩码的值是固定的，用于检测边缘，不需要通过训练来调整

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            #nn.Parameter(torch.FloatTensor(bias))转换为一个 torch.FloatTensor 类型的张量，然后将其包装为 nn.Parameter 类型
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # 拉普拉斯算子掩码
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias对y0进行填充 周围填充1个像素的空白
            #之所以要有这个padding操作 应该是为保证后面3*3的卷积的输出大小一致
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            #为了将self.b0张量的形状从一维转换为四维，以适应后续的广播操作 形状变为 (1, self.out_planes, 1, 1)
            #第一个 1 代表批量维度 因为偏置是作用于输出的特征图的，而每个特征图的维度是(Batch size,channels,H,W)
            #-1 会自适应为channels，在这里就是out_planes，其中批次1也会因为广播机制进行适应
            b0_pad = self.b0.view(1, -1, 1, 1)
            #对y0的边界加上偏置
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
            #每个输出通道只会用一个单通道的 3x3 的卷积核对输入特征图的相应通道进行卷积操作
        return y1

    #重参数化操作
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:#cpu
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            # self.k0 = conv0.weight  (self.mid_planes，self.inp_planes，1，1)
            # self.k1 = conv1.weight  (self.out_planes，self.mid_planes，3，3) 输入特征图NCHW
            #               weight =  (self.inp_planes,self.mid_planes，1，1) 卷积核inp_planes,mid_planes,H,W
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))#out_planes inp_planes 3 3
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            #self.b0 扩展为一个形状为 (1, self.mid_planes, 3, 3) 的b0,在进行k0，b0的卷积的时候，得到的输出特征图就是mid_planes通道，所以b0偏置的通道数是mid_planes
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1
            #input (1, self.mid_planes, 3, 3) NCHW
            #weight(self.out_planes，self.mid_planes，3，3) out_planes,mid_planes,H,W
            #RB 1 out_planes 1 1--->out_planes一维
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class ECB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt=False, deploy=False):
        #act_type 激活函数类型
        #with_idt 表示是否在网络中加入恒等连接（Skip Connection）
        #deploy 在部署模式下，网络被简化为一个单独的卷积层，这有助于减少计算量并在实际应用中加速推理过程。
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.deploy = deploy

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False
        '''
        一般的恒等连接（Identity Connection），也称为跳跃连接（Skip Connection）
        在恒等连接中，输入信号可以直接绕过一个或多个层，并与这些层的输出相加。具体来说，假设有一个输入 𝑥，
        经过一些层的处理后输出为 𝐹(𝑥)，那么在使用恒等连接的情况下，输出将变为 𝐹(𝑥)+𝑥。
        
        '''

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
            #self.rbr_reparam是一个3x3的标准卷积层,只初始化一个3x3的卷积层
        else:
            #非部署模式下，初始化多个卷积层，每个卷积层执行不同的操作
            self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
            self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
            self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
            self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
            self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)#inplace=True 使得 ReLU 激活函数直接修改输入张量
        elif self.act_type == 'lrelu':
            self.act = nn.LeakyReLU()
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        # \ 是行连接符，用于将一行代码拆分成多行
        y = self.conv3x3(x) + \
            self.conv1x1_3x3(x) + \
            self.conv1x1_sbx(x) + \
            self.conv1x1_sby(x) + \
            self.conv1x1_lpl(x)
        if self.with_idt:
            y += x
        return self.act(y)

    #将模型从训练模式转换为部署模式
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):#检查模型是否已经处于部署模式
            return
        #self.rep_params() 获取合并后的卷积核和偏置
        RK, RB = self.rep_params()
        self.rbr_reparam = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.rbr_reparam.weight.data = RK
        self.rbr_reparam.bias.data = RB
        #删除这些属性 删除这些属性的操作是为了将模型从训练模式转换为部署模式。删除原有的多个卷积层，只保留一个合并后的卷积层 rbr_reparam
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1_3x3')
        self.__delattr__('conv1x1_sbx')
        self.__delattr__('conv1x1_sby')
        self.__delattr__('conv1x1_lpl')
        self.deploy = True

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0 + K1 + K2 + K3 + K4), (B0 + B1 + B2 + B3 + B4)

        #进行恒等连接
        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


if __name__ == '__main__':
    # # test seq-conv
    x = torch.randn(1, 3, 480, 620).cuda()
    conv = SeqConv3x3('conv1x1-conv3x3', 3, 3, 2).cuda()
    y0 = conv(x)
    RK, RB = conv.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print(torch.mean(y0 - y1))
    #计算 y0 和 y1 之间的均值差异。如果差异接近零，则说明 SeqConv3x3 的实现和重参数化后的卷积操作是一致的

    # test ecb
    x = torch.randn(1, 3, 480, 620).cuda()
    act_type = 'prelu'
    ecb = ECB(3, 64, 2, act_type=act_type, with_idt=True).cuda()
    y0 = ecb(x)

    RK, RB = ecb.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    if act_type == 'prelu':
        act = nn.PReLU(num_parameters=64).cuda()
        y1 = act(y1)
    elif act_type == 'relu':
        act = nn.ReLU(inplace=True).cuda()
        y1 = act(y1)
    elif act_type == 'rrelu':
        act = nn.RReLU(lower=-0.05, upper=0.05).cuda()
        y1 = act(y1)
    elif act_type == 'softplus':
        act = nn.Softplus().cuda()
        y1 = act(y1)
    elif act_type == 'linear':
        pass
    print(torch.mean(y0 - y1))



