import torch
import torch.nn.functional as F
from utils.pyramids import *
from losses.discriminator_loss import DiscriminatorLoss
from losses.discriminator_loss import adversarial_loss
from tqdm import tqdm
from utils.pyramids import GaussianPyramid
from losses.losses import GeneratorLoss, DiscLoss


def evaluate(epoch, net, net_D, dataloader, device, ps):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss_generator = 0
    discriminator_loss = DiscLoss()
    gen_loss = GeneratorLoss(size=ps)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        exp_images, gt_images = batch['exp_image'], batch['gt_image']

        with torch.no_grad():
            # Predictions (forward propagation)
            _, y_pred = net(exp_images)
            GP = GaussianPyramid(4) 
            G_pyramid = GP(gt_images)
            
            for pyramid in G_pyramid:
                G_pyramid[pyramid] = G_pyramid[pyramid].to(device=device, dtype=torch.float32)

            laplacian_pyr, y_pred = net(exp_images)

            # Critertions for Losses:
            mae_loss = nn.L1Loss()
            bcelog_loss = nn.BCEWithLogitsLoss()  # This already includes sigmoid

            if (epoch + 1 >= 15) and (ps == 256):

                # Adversarial Loss (only for 256 patches
                disc_adv = net_D(y_pred['subnet_16'])
                adv_loss = bcelog_loss(disc_adv, torch.ones_like(disc_adv))

            else:
                adv_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)

            # Generator Loss
            loss_generator = 4 * mae_loss(y_pred['subnet_24_1'],
                                          F.interpolate(G_pyramid['level4'], (y_pred['subnet_24_1'].shape[2],
                                                                              y_pred['subnet_24_1'].shape[3]),
                                                        mode='bilinear', align_corners='True')) + \
                             2 * mae_loss(y_pred['subnet_24_2'],
                                          F.interpolate(G_pyramid['level3'], (y_pred['subnet_24_2'].shape[2],
                                                                              y_pred['subnet_24_2'].shape[3]),
                                                        mode='bilinear', align_corners='True')) + \
                             mae_loss(y_pred['subnet_24_3'],
                                      F.interpolate(G_pyramid['level2'], (y_pred['subnet_24_3'].shape[2],
                                                                          y_pred['subnet_24_3'].shape[3]),
                                                    mode='bilinear', align_corners='True')) + \
                             mae_loss(y_pred['subnet_16'], G_pyramid['level1']) + adv_loss

            val_loss_generator += loss_generator

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return val_loss_generator
    return val_loss_generator / num_val_batches