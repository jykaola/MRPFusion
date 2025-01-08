class easyFusion(nn.Module):
    def __init__(self, deploy=False):
        super(easyFusion, self).__init__()
        ch = [1, 16, 32, 64, 128]
        self.act_type = 'lrelu'

        # Initial convolution layers
        self.conv0_vi = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)
        self.conv0_ir = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)

        # RepBlock layers
        self.conv1_vi = RepBlock(ch[1], ch[2], 2, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv1_ir = RepBlock(ch[1], ch[2], 2, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2_vi = RepBlock(ch[2], ch[3], 2, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2_ir = RepBlock(ch[2], ch[3], 2, act_type=self.act_type, with_idt=False, deploy=deploy)

        # Attention modules
        self.CBAM1 = CBAMLayer(ch[3])
        self.CBAM2 = CBAMLayer(ch[3])

        # Decoder layers (no skip connections)
        self.conv1 = RepBlock(ch[4], ch[3], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2 = RepBlock(ch[3], ch[2], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv3 = RepBlock(ch[2], ch[1], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv4 = nn.Conv2d(ch[1], ch[0], kernel_size=1, padding=0)

        # Activation functions
        self.act = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, image_vi, image_ir):
        # Encoder part
        x_vi = self.act(self.conv0_vi(image_vi))
        x_ir = self.act(self.conv0_ir(image_ir))

        # First RepBlock layer
        x_vi = self.conv1_vi(x_vi)
        x_ir = self.conv1_ir(x_ir)

        # Second RepBlock layer
        x_vi = self.conv2_vi(x_vi)
        x_ir = self.conv2_ir(x_ir)

        # Fusion
        x = torch.cat([(x_vi * x_ir), (self.CBAM1(x_vi) + self.CBAM2(x_ir))], dim=1)

        # Decoder part (no skip connections)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.tanh(self.conv4(x))

        return x / 2 + 0.5  # Scale output to [0, 1]