import torch.nn as nn
import torch

# class Pyr_Loss (nn.Module):
#     def __init__(self, weight=1.0):
#         super(Pyr_Loss, self).__init__()
#         self.weight =weight
#         self.criterion = nn.L1Loss(reduction='sum')
#     def forward(self,Y_list, T_list):
#         n = len(Y_list)
#         loss = 0
#         for m in range(0,n-1):
#             loss += self.weight*(2**(n-m-2))*self.criterion(Y_list[m],F.interpolate(T_list[m],(Y_list[m].shape[2],Y_list[m].shape[3]),mode='bilinear',align_corners=True))/Y_list[m].shape[0]
#         return loss

class DiscriminatorLoss(nn.Module):
    def __init__(self, net_d, device):
        super(DiscriminatorLoss, self).__init__()
        self.net_d = net_d
        self.device = device

    def forward(self, t, y):

        #ps = t.size(dim=2)
        epsilon = torch.tensor([[10e-09]], requires_grad=False).to(device=self.device, dtype=torch.float32)
        loss_real = -torch.mean(torch.log(torch.maximum(self.net_d(t), epsilon)))
        loss_generated = -torch.mean(torch.log(torch.maximum(1-self.net_d(y).detach(), epsilon)))
        #disc_loss = loss_real + loss_generated

        return loss_real, loss_generated


def adversarial_loss(net_d, y, device):
    ps = y.size(dim=2)
    epsilon = torch.tensor([[10e-09]], requires_grad=False).to(device=device, dtype=torch.float32)
    W = (ps**2)*12
    adv_loss = -W*torch.mean(torch.log(torch.maximum(net_d(y), epsilon)))
    #adv_loss = (12*ps*ps)*torch.mean(adv_loss)

    return adv_loss


#def sigmoid(z):
#    return 1 / (1 + torch.exp(-z))