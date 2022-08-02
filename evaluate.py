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

            y_pred = [y for y in y_pred.values()]
            g_pyramid = [t for t in G_pyramid.values()]

            if (epoch + 1 >= 15) and (ps == 256):

                y_pred_2 = [Y.detach() for Y in y_pred]
                disc_fake = net_D(y_pred_2[-1])

                disc_real = net_D(g_pyramid[-1])

                disc_loss = discriminator_loss(disc_fake, disc_real)
                rec_loss, pyr_loss, adv_loss, loss_generator = gen_loss(y_pred, g_pyramid, disc_fake,
                                                                        withoutadvloss=False)


            else:

                rec_loss, pyr_loss, loss_generator = gen_loss(y_pred, g_pyramid, withoutadvloss=True)


            val_loss_generator += loss_generator

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return val_loss_generator
    return val_loss_generator / num_val_batches