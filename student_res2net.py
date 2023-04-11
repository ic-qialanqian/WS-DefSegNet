import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch.nn.init as init
from torch.nn import Conv2d, Parameter, Softmax
from functools import partial
import matplotlib.pyplot as plt
import math
from DTEN import DTEN
from Res2Net_v1b import res2net50_v1b_26w_4s




    
class student_net(nn.Module):
    def __init__(self):
        super(student_net, self).__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
       
        
                 
                 
        self.DTEN = DTEN(in_channels=[2048,2048,2048],out_channels=256,num_outs=5,
                 hidden_dim=256,
                 position_embedding='sine',) 
                 
                 
        self.initial_out = nn.Conv2d(2048, 1, kernel_size=1, padding=0)
        
        
        

        self.transformer_ouput = nn.Conv2d(256, 1, kernel_size=1, padding=0)
        
        
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x,mask):
        input = x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)
        
        merge3 = F.interpolate(x4, 22, mode='bilinear')
        merge4 = F.interpolate(x4, 44, mode='bilinear')
        merge5 = F.interpolate(x4, 88, mode='bilinear')
        dten_in = [merge3,merge4,merge5]
        
        dten_out = self.DTEN(dten_in,mask)
        
        res2net_out = F.interpolate(self.initial_out(x4), input.size()[2:], mode="bilinear")

        
        output_final = F.interpolate(self.transformer_ouput(dten_out), input.size()[2:], mode="bilinear")

        if self.training:
            return res2net_out,output_final

        return output_final

    


if __name__ == "__main__":
    model = student_net()
    model.cuda()
    input = torch.autograd.Variable(torch.zeros(4, 3, 256, 256)).cuda()
    mask = torch.autograd.Variable(torch.zeros(4, 1, 256, 256)).cuda()
    output = model(input, mask)
   
