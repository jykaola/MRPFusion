# class easyFusion(nn.Module):
#     def __init__(self):
#         super(easyFusion, self).__init__()
#         ch = [1, 16, 32, 64, 128]
#         self.act_type = 'lrelu'
#
#         # 初始层（没有跳跃连接和 RMFB，仅 3x3 卷积） 第一层是1*1卷积
#         self.conv0_vi = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)
#         self.conv0_ir = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)
#
#         # 替换 RepBlock 为普通 3x3 卷积
#         self.conv1_vi = nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1)
#         self.conv1_ir = nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1)
#         self.conv2_vi = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1)
#         self.conv2_ir = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1)
#
#         # 解码器层，同样使用普通 3x3 卷积
#         self.conv1 = nn.Conv2d(ch[3] * 2, ch[3], kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(ch[3], ch[2], kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(ch[2], ch[1], kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(ch[1], ch[0], kernel_size=1, padding=0)
#
#         # 激活函数
#         self.act = nn.LeakyReLU()
#         self.tanh = nn.Tanh()
#
#         self.CBAM1 = CBAMLayer(ch[3])
#         self.CBAM2 = CBAMLayer(ch[3])
#
#     def forward(self, image_vi, image_ir):
#         # 编码器部分
#         x_vi = self.act(self.conv0_vi(image_vi))
#         x_ir = self.act(self.conv0_ir(image_ir))
#
#         # 卷积层（替换掉 RepBlock）
#         x_vi = self.act(self.conv1_vi(x_vi))
#         x_ir = self.act(self.conv1_ir(x_ir))
#
#         x_vi = self.act(self.conv2_vi(x_vi))
#         x_ir = self.act(self.conv2_ir(x_ir))
#
#         # 融合
#         # x = torch.cat([x_vi, x_ir], dim=1)  # 没有 CBAM，直接通道拼接
#         x = torch.cat([(x_vi * x_ir), (self.CBAM1(x_vi) + self.CBAM2(x_ir))], dim=1)
#
#         # 解码器部分
#         x = self.act(self.conv1(x))
#         x = self.act(self.conv2(x))
#         x = self.act(self.conv3(x))
#         x = self.tanh(self.conv4(x))
#
#         return x / 2 + 0.5  # 输出范围从 [-1, 1] 变换到 [0, 1]