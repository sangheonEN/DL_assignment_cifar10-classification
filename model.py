import torch.nn as nn
from backbone.res import resnet
import torchvision.models.mobilenetv3 as mobilenet
import torchvision.models.efficientnet as efficientnet

class ResNet(nn.Module):
    def __init__(self, class_num):
        super(ResNet, self).__init__()
        self.class_num = class_num
        self.model = resnet.resnext50_32x4d(pretrained=True)
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        # feature shape: 512*4, classnum

        feature = self.model(x)
        output = self.fc(feature)

        return output

class Efficientnet(nn.Module):
    def __init__(self, class_num):
        super(Efficientnet, self).__init__()
        self.class_num = class_num
        self.model = efficientnet.efficientnet_b4(pretrained=True)
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        feature = self.model(x)
        output = self.fc(feature)

        return output
#
class MobileNetV3(nn.Module):
    def __init__(self, class_num):
        super(MobileNetV3, self).__init__()
        self.class_num = class_num
        self.model = mobilenet.mobilenet_v3_large(pretrained=True)
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        feature = self.model(x)
        output = self.fc(feature)

        return output
