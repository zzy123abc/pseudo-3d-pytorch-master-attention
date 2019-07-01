import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionModule(nn.Module):

    def __init__(self):
        super(AttentionModule, self).__init__()

    def forward(self, x):
        x1 = x.contiguous().view(x.size()[0],2048,-1)
        x2 = x1.transpose(1,2)
        x3 = torch.matmul(x2, x1)
        x3 = F.softmax(x3,dim=-1)
        x4 = x1
        x4= torch.matmul(x4,x3)
        x4 =x4.view(x.size())

        return x4
