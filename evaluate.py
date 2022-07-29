import torch
import torch.nn.functional as F
from utils.pyramids import *
from losses.losses import DiscriminatorLoss
from losses.losses import GeneratorLoss
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

            if (epoch + 1 >= 11) and (ps == 256):
                generator_loss = GeneratorLoss(net_d=net_D, device=device, use_adv=True)
                loss_generator, adv_loss = generator_loss(y_pred, G_pyramid)

            else:
                generator_loss = GeneratorLoss(net_d=net_D, device=device, use_adv=False)
                loss_generator, adv_loss = generator_loss(y_pred, G_pyramid)

            # Generator Loss
            val_loss_generator += loss_generator + val_loss_generator

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return val_loss_generator
    return val_loss_generator / num_val_batches