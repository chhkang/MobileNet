import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.conv = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.model = nn.Sequential(
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),  # 8
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),  # 4
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2), #2
            conv_dw(512,512,1),
            conv_dw(512,512,1),
            conv_dw(512,512,1),
            conv_dw(512,512,1),
            conv_dw(512,512,1),
            conv_dw(512, 1024, 1),
            conv_dw(1024,1024, 1)
        )
        self.avgpool = nn.AvgPool2d(2)
        self.linear = nn.Linear(1024,100)
    def forward(self,x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.model(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out