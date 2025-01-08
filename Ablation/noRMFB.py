class easyFusion(nn.Module):
    def __init__(self):
        super(easyFusion, self).__init__()
        ch = [1, 16, 32, 64, 128]
        self.act_type = 'lrelu'

        # Initial layers (encoder)
        self.conv0_vi = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)
        self.conv0_ir = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)

        # Middle convolution layers to replace RMFB
        self.conv1_vi = nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1)
        self.conv1_ir = nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1)
        self.conv2_vi = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1)
        self.conv2_ir = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1)

        # Channel adjustment layers for skip connections
        self.adjust_skip2 = nn.Conv2d(ch[2], ch[4], kernel_size=1, padding=0)
        self.adjust_skip1 = nn.Conv2d(ch[1], ch[3], kernel_size=1, padding=0)

        # Decoder layers
        self.conv1 = nn.Conv2d(ch[3] * 2, ch[3], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch[3], ch[2], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ch[2], ch[1], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(ch[1], ch[0], kernel_size=1, padding=0)

        # Activation functions
        self.act = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # Attention modules
        self.CBAM1 = CBAMLayer(ch[3])
        self.CBAM2 = CBAMLayer(ch[3])

    def forward(self, image_vi, image_ir):
        # Encoder
        x_vi = self.act(self.conv0_vi(image_vi))
        x_ir = self.act(self.conv0_ir(image_ir))

        # Save first skip connection
        skip1_vi = x_vi
        skip1_ir = x_ir

        # First convolution layer
        x_vi = self.act(self.conv1_vi(x_vi))
        x_ir = self.act(self.conv1_ir(x_ir))

        # Save second skip connection
        skip2_vi = x_vi
        skip2_ir = x_ir

        # Second convolution layer
        x_vi = self.act(self.conv2_vi(x_vi))
        x_ir = self.act(self.conv2_ir(x_ir))

        # Fusion
        x = torch.cat([(x_vi * x_ir), (self.CBAM1(x_vi) + self.CBAM2(x_ir))], dim=1)

        # Decoder with skip connections
        x = self.act(self.conv1(x + self.adjust_skip2(skip2_vi + skip2_ir)))  # Adjusted skip2
        x = self.act(self.conv2(x + self.adjust_skip1(skip1_vi + skip1_ir)))  # Adjusted skip1
        x = self.act(self.conv3(x))
        x = self.tanh(self.conv4(x))

        return x / 2 + 0.5  # Scale output to [0, 1]