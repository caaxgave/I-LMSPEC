import torch.nn as nn
import torch
import torch.nn.functional as F


class GeneratorLoss(nn.Module):
    def __init__(self, net_d, device, use_adv=False):
        super(GeneratorLoss, self).__init__()
        self.net_d = net_d
        self.criterion = nn.L1Loss()
        self.device = device
        self.use_adv = use_adv

        if self.use_adv:
            self.adv_loss = AdversarialLoss(self.net_d)
        else:
            self.adv_loss = torch.tensor([[0]]).to(device=self.device, dtype=torch.float32)

    def forward(self, y, t):

        if self.use_adv:
            self.adv_loss = self.adv_loss(y['subnet_16'])

        # Generator Loss
        loss_generator = 4 * self.criterion(y['subnet_24_1'],
                                      F.interpolate(t['level4'],
                                                    (y['subnet_24_1'].shape[2],
                                                     y['subnet_24_1'].shape[3]),
                                                    mode='bilinear', align_corners=True)) + \
                         2 * self.criterion(y['subnet_24_2'],
                                      F.interpolate(t['level3'],
                                                    (y['subnet_24_2'].shape[2],
                                                     y['subnet_24_2'].shape[3]),
                                                    mode='bilinear', align_corners=True)) + \
                         self.criterion(y['subnet_24_3'],
                                  F.interpolate(t['level2'],
                                                (y['subnet_24_3'].shape[2],
                                                 y['subnet_24_3'].shape[3]),
                                                mode='bilinear', align_corners=True)) + \
                         self.criterion(y['subnet_16'], t['level1']) + self.adv_loss

        return loss_generator, adv_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, net_d, device, use_adv=False):
        super(DiscriminatorLoss, self).__init__()
        self.net_d = net_d
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device
        self.use_adv = use_adv

    def forward(self, y, t):

        if self.use_adv:
            #real_loss = -torch.mean(torch.log(torch.sigmoid(self.net_d(t['level1'])) + 1e-9))
            disc_real = self.net_d(t['level1'])
            real_loss = self.criterion(disc_real + 1e-9, torch.ones_like(disc_real))
            #fake_loss = -torch.mean(torch.log(1 - torch.sigmoid(self.net_d(y['subnet_16'].detach())) + 1e-9))
            disc_fake = self.net_d(y['subnet_16'])
            fake_loss = self.criterion(disc_fake + 1e-9, torch.zeros_like(disc_fake))
        else:
            real_loss = torch.tensor([[0]]).to(device=self.device, dtype=torch.float32)
            fake_loss = torch.tensor([[0]]).to(device=self.device, dtype=torch.float32)

        return real_loss, fake_loss


class AdversarialLoss(nn.Module):
    def __init__(self, net_d):
        super(AdversarialLoss, self).__init__()
        self.net_d = net_d
        self.criterion_adv = nn.BCEWithLogitsLoss()

    def forward(self, y):
        #W = 12*(y.shape[2] ** 2)
        #adv_loss = -W * torch.mean(torch.log(torch.sigmoid(y) + 1e-9))
        disc_adv = self.net_d(y)
        adv_loss = self.criterion_adv(disc_adv, torch.ones_like(disc_adv))

        return adv_loss
