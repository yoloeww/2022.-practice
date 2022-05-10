import torchvision 
import torch
from torchvision import models
dir(models)
alexnet=models.AlexNet()
resnet=models.resnet101(pretrained=True)
