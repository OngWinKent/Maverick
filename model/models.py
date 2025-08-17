"""
Source: https://github.com/weiaicunzai/pytorch-cifar100
"""

from torch import nn
import torch
from model.resnet import ResNet, BasicBlock, BottleNeck
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ResNet18(num_classes, input_channels):
    """return a ResNet 18 object"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channel= input_channels)

def ResNet34(num_classes, input_channels):
    """return a ResNet 34 object"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes= num_classes, input_channel= input_channels)

def ResNet50(num_classes, input_channels):
    """return a ResNet 50 object"""
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes= num_classes, input_channel= input_channels)

def ResNet101(num_classes, input_channels):
    """return a ResNet 101 object"""
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes= num_classes, input_channel= input_channels)

def ResNet152(num_classes, input_channels):
    """return a ResNet 152 object"""
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes= num_classes, input_channel= input_channels)