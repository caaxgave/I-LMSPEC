import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from torchvision.models.vgg import vgg16
import numpy as np


class PyrLoss (nn.Module):
    def __init__(self,weight=1.0):
        super(PyrLoss, self).__init__()
        self.weight =weight
        self.criterion = nn.L1Loss(reduction='sum')

    def forward(self,Y_list, T_list):
        n = len(Y_list)
        loss = 0
        print(Y_list.shape())
        print(T_list.shape())
        for m in range(0, n-1):
            loss += self.weight*(2**(n-m-2)) *\
                    self.criterion(Y_list[m], F.interpolate(T_list[m],(Y_list[m].shape[2],Y_list[m].shape[3]),
                                                            mode='bilinear',align_corners=True))/Y_list[m].shape[0]
        return loss


class RecLoss(nn.Module):
    def __init__(self,weight=1):
        super(RecLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss(reduction='sum')

    def forward(self, Y_list, T_list):
        loss = self.weight * self.criterion(Y_list[-1],T_list[-1])/Y_list[-1].shape[0]
        return loss


class AdvLoss(nn.Module):
    def __init__(self, size=256, weight=1.0):
        super(AdvLoss,self).__init__()
        self.weight = weight
        self.size = size

    def forward(self, P_Y):
        loss = -self.weight * 12 *self.size*self.size*torch.mean(torch.log(torch.sigmoid(P_Y)+1e-9))
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self, size=256, Pyr_weight=1.0, Rec_weight=1.0, Adv_weight=1.0):
        super(GeneratorLoss,self).__init__()
        self.pyr_loss = PyrLoss(Pyr_weight)
        self.rec_loss = RecLoss(Rec_weight)
        self.adv_loss = AdvLoss(size,Adv_weight)

    def forward(self, Y_list, T_list, P_Y=None, withoutadvloss=False):
        pyrloss =self.pyr_loss(Y_list, T_list)
        recloss =self.rec_loss(Y_list,T_list)
        if withoutadvloss:
            myloss = pyrloss + recloss
            return recloss,pyrloss,myloss
        else:
            advloss =self.adv_loss(P_Y)
            myloss = pyrloss + recloss + advloss
            return recloss, pyrloss, advloss, myloss


class DiscLoss(nn.Module):
    def __init__(self):
        super(DiscLoss,self).__init__()

    def forward(self, P_Y, P_T):
        loss = -torch.mean(torch.log(torch.sigmoid(P_T) + 1e-9)) - torch.mean(torch.log(1 - torch.sigmoid(P_Y) + 1e-9))
        return loss


if __name__ =='__main__':
    a = torch.rand([2,3,32,32]).float()
    p = torch.rand([2,1]).float()
    alist = [a]
    total_loss = GeneratorLoss()
    t = total_loss(alist,alist,p)
    print(t)