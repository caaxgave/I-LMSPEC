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

            # Losses:
            # Generator Loss
            mae_loss = nn.L1Loss(reduction='sum')

            # bce_loss = nn.BCELoss()
            bcelog_loss = nn.BCEWithLogitsLoss()  # This already includes sigmoid

            if (epoch+1 >= 15) and (ps == 256):

                #Adversarial Loss (only for 256 patches)
                disc_fake = net_D(y_pred['subnet_16'])
                fake_loss = bcelog_loss(disc_fake, torch.zeros_like(disc_fake))
                disc_real = net_D(G_pyramid['level1'])
                real_loss = bcelog_loss(disc_real, torch.ones_like(disc_real))
                disc_loss = (fake_loss + real_loss) #/ 2

                disc_adv = net_D(y_pred['subnet_16'])
                adv_loss = bcelog_loss(disc_adv, torch.ones_like(disc_adv))

                # adv_loss = adversarial_loss(net_D, y_pred['subnet_16'], device=device)

            else:
            #     #disc_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)
            #     #real_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)
            #     #fake_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)
                 adv_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)

            # Generator Loss
            val_loss_generator += (
                    4 * mae_loss(y_pred['subnet_24_1'], G_pyramid['level3']) +
                    2 * mae_loss(y_pred['subnet_24_2'], G_pyramid['level2']) +
                    mae_loss(y_pred['subnet_24_3'], G_pyramid['level1']) +
                    mae_loss(y_pred['subnet_16'], G_pyramid['level1'])) + adv_loss

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return val_loss_generator
    return val_loss_generator / num_val_batches