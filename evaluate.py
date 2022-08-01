import torch
import torch.nn.functional as F
from utils.pyramids import *
from losses.discriminator_loss import DiscriminatorLoss
from losses.discriminator_loss import adversarial_loss
from tqdm import tqdm
from utils.pyramids import GaussianPyramid


def evaluate(epoch, net, net_D, dataloader, device, ps):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss_generator = 0

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

            mae_loss = nn.L1Loss(reduction='sum')
            # bcelog_loss = nn.BCEWithLogitsLoss()

            if (epoch + 1 >= 15) and (ps == 256):

                adv_loss = adversarial_loss(net_D, y_pred['subnet_16'], device)

            else:

                adv_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)

            # Generator Loss
            C = (ps ** 2) * 3
            W = (ps ** 2) * 12
            val_loss_generator += (1/C)*((4 * mae_loss(y_pred['subnet_24_1'],
                                              F.interpolate(G_pyramid['level4'],
                                                            (y_pred['subnet_24_1'].shape[2],
                                                             y_pred['subnet_24_1'].shape[3]))) +
                                 2 * mae_loss(y_pred['subnet_24_2'],
                                              F.interpolate(G_pyramid['level3'],
                                                            (y_pred['subnet_24_2'].shape[2],
                                                             y_pred['subnet_24_2'].shape[3]))) +
                                 mae_loss(y_pred['subnet_24_3'], F.interpolate(G_pyramid['level2'],
                                                            (y_pred['subnet_24_3'].shape[2],
                                                             y_pred['subnet_24_3'].shape[3]))) +
                                 mae_loss(y_pred['subnet_16'], F.interpolate(G_pyramid['level1'],
                                                            (y_pred['subnet_16'].shape[2],
                                                             y_pred['subnet_16'].shape[3]))))/y_pred['subnet_16'].shape[0] \
                                 + W*adv_loss)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return val_loss_generator
    return val_loss_generator / num_val_batches