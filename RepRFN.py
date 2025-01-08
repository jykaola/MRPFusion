from torch import nn
import torch
import torch.nn.functional as F
from itertools import repeat
import collections.abc


# def conv_layer(in_channels, out_channels, kernel_size, stride=1):
#     # kernel_size参数预处理
#     if not isinstance(kernel_size, collections.abc.Iterable):
#         kernel_size = tuple(repeat(kernel_size, 2))
#     padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
#     return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#
#
# def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
#     act_type = act_type.lower()
#     if act_type == 'relu':
#         act_func = nn.ReLU(inplace)
#     elif act_type == 'lrelu':
#         act_func = nn.LeakyReLU(neg_slope, inplace)
#     elif act_type == 'prelu':
#         act_func = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
#     # TODO: 新增silu和gelu激活函数
#     elif act_type == 'silu':
#         pass
#     elif act_type == 'gelu':
#         pass
#     else:
#         raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
#     return act_func


# thanks for ECBSR: https://github.com/xindongzhang/ECBSR
class SeqConv3x3(nn.Module):
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

        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

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
            # init mask
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
            return y1
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
            return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
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


class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels,depth_multiplier, act_type='prelu', with_idt=False,deploy=False):
        super(RepBlock, self).__init__()

        self.deploy = deploy
        self.depth_multiplier = depth_multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_type = act_type

        if with_idt and (self.in_channels == self.out_channels):
            self.with_idt = True
        else:
            self.with_idt = False

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
            self.rbr_3x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                            stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                            stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_conv1x1_3x3_branch = SeqConv3x3('conv1x1-conv3x3', self.in_channels, self.out_channels, self.depth_multiplier)
            self.rbr_conv1x1_sbx_branch = SeqConv3x3('conv1x1-sobelx', self.in_channels, self.out_channels,-1)
            self.rbr_conv1x1_sby_branch = SeqConv3x3('conv1x1-sobely', self.in_channels, self.out_channels,-1)

            self.rbr_conv1x1_lpl_branch = SeqConv3x3('conv1x1-laplacian', self.in_channels, self.out_channels,-1)
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_channels)
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

    def forward(self, inputs):
        if (self.deploy):
            return self.act(self.rbr_reparam(inputs))
        else:
            y=self.rbr_3x3_branch(inputs)+self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(inputs)
            + self.rbr_1x1_branch(inputs) +self.rbr_conv1x1_3x3_branch(inputs)   + self.rbr_conv1x1_sbx_branch(inputs)
            + self.rbr_conv1x1_sby_branch(inputs) + self.rbr_conv1x1_lpl_branch(inputs)
        if self.with_idt:
            y += x
        return self.act(y)

    def switch_to_deploy(self):
        if hasattr(self, 'rep_param'):#检查模型是否已经处于部署模式
            return
        kernel, bias = self.rep_params()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_3x3_branch')
        self.__delattr__('rbr_3x1_branch')
        self.__delattr__('rbr_1x3_branch')
        self.__delattr__('rbr_1x1_branch')
        self.__delattr__('rbr_conv1x1_3x3_branch')
        self.__delattr__('rbr_conv1x1_sbx_branch')
        self.__delattr__('rbr_conv1x1_sby_branch')
        self.__delattr__('rbr_conv1x1_lpl_branch')
        self.deploy = True

    def rep_params(self):
        # 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data

        # 1x1 1x3 3x1 branch
        kernel_1x1_1x3_3x1_fuse, bias_1x1_1x3_3x1_fuse = self._fuse_1x1_1x3_3x1_branch(self.rbr_1x1_branch,
                                                                                       self.rbr_1x3_branch,
                                                                                       self.rbr_3x1_branch)
        # 1x1+3x3 branch
        kernel_1x1_3x3,bias_1x1_3x3 = self.rbr_conv1x1_3x3_branch.rep_params()
        kernel_1x1_sbx, bias_1x1_sbx = self.rbr_conv1x1_sbx_branch.rep_params()
        kernel_1x1_sby, bias_1x1_sby = self.rbr_conv1x1_sby_branch.rep_params()
        kernel_1x1_lpl, bias_1x1_lpl = self.rbr_conv1x1_lpl_branch.rep_params()
        RK,RB=(kernel_3x3+kernel_1x1_1x3_3x1_fuse+kernel_1x1_3x3+kernel_1x1_sbx+kernel_1x1_sby+kernel_1x1_lpl),(bias_3x3+bias_1x1_1x3_3x1_fuse+bias_1x1_sbx+bias_1x1_sby+bias_1x1_lpl+bias_1x1_3x3)
        # 进行恒等连接
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

    def _fuse_1x1_1x3_3x1_branch(self, conv1, conv2, conv3):
        weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (0, 0, 1, 1)) + F.pad(
            conv3.weight.data, (1, 1, 0, 0))
        bias = conv1.bias.data + conv2.bias.data + conv3.bias.data
        return weight, bias

    def _fuse_1x1_3x3_branch(self, conv1, conv2):
        weight = F.conv2d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3))
        return weight

if __name__ == '__main__':
    # # test seq-conv
    x = torch.randn(1, 3, 480, 620).cuda()
    conv = SeqConv3x3('conv1x1-conv3x3', 3, 3, 2).cuda()
    y0 = conv(x)
    RK, RB = conv.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print(torch.mean(y0 - y1))
    #计算 y0 和 y1 之间的均值差异。如果差异接近零，则说明 SeqConv3x3 的实现和重参数化后的卷积操作是一致的

    # test
    x = torch.randn(1, 3, 480, 620).cuda()
    act_type = 'prelu'
    rbk = RepBlock(3, 64, 2, act_type=act_type, with_idt=False).cuda()
    y0 = rbk(x)
    print(y0.shape)
    RK, RB = rbk.rep_params()
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
