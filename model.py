import torch.nn as nn
import torch.nn.functional as F

class DWConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(DWConv2d,self).__init__()
        self.DWConv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride= stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.PWConv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride = 1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        out = F.relu(self.bn1(self.DWConv(x)))
        out = F.relu(self.bn2(self.PWConv(out)))
        return out

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.conv = nn.Conv2d(3,32,3,2,1) #16
        self.bn = nn.BatchNorm2d(32)
        self.model = nn.Sequential(
            DWConv2d(32, 64, 1),
            DWConv2d(64, 128, 2),  # 8
            DWConv2d(128, 128, 1),
            DWConv2d(128, 256, 2),  # 4
            DWConv2d(256, 256, 1),
            DWConv2d(256, 512, 2), #2
            DWConv2d(512,512,1),
            DWConv2d(512,512,1),
            DWConv2d(512,512,1),
            DWConv2d(512,512,1),
            DWConv2d(512,512,1),
            DWConv2d(512, 1024, 1)
        )
        self.avgpool = nn.AvgPool2d(2)
        self.linear = nn.Linear(1024,100)
        return nn.Sequential(*list)
    def forward(self,x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.model(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out