from unet import UNet24
from unet import UNet16
from utils.pyramids import *
import torch


class Generator(nn.Module):
    def __init__(self, n_channels, device, bilinear=False):
        super(Generator, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.device = device
        
        # Calling pyramid functions:
        self.LP = LaplacianPyramid(4)
        
        # defining UNet sub-networks
        self.unet24 = UNet24(n_channels=self.n_channels, bilinear=self.bilinear)
        self.unet24_res = UNet24(n_channels=self.n_channels, res_layer=True, bilinear=self.bilinear)
        self.unet16 = UNet16(n_channels=self.n_channels, res_layer=True, bilinear=self.bilinear)
        
    def forward(self, x):

        L_pyramid = self.LP(x)
        
        for pyramid in L_pyramid:
            L_pyramid[pyramid] = L_pyramid[pyramid].to(device=self.device, dtype=torch.float32)
        
        x0 = L_pyramid['level4']
        y_hat0 = self.unet24(x0)
        x1 = torch.add(y_hat0, L_pyramid['level3'])
        y_hat1 = self.unet24_res(x1)
        x2 = torch.add(y_hat1, L_pyramid['level2'])
        y_hat2 = self.unet24_res(x2)
        x3 = torch.add(y_hat2, L_pyramid['level1'])
        y_hat3 = self.unet16(x3)

        outputs = {'subnet_24_1': y_hat0,
                   'subnet_24_2': y_hat1,
                   'subnet_24_3': y_hat2,
                   'subnet_16': y_hat3}
        
        return outputs