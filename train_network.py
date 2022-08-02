import logging
from pathlib import Path
import os
import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
from utils.data_loading import ImageDataset
from utils.pyramids import GaussianPyramid
from losses.discriminator_loss import DiscriminatorLoss
from losses.discriminator_loss import adversarial_loss
from evaluate import evaluate
from contextlib import contextmanager
from losses.losses import GeneratorLoss, DiscLoss
import warnings


def train_net(net,
              net_D,
              drop_rate,
              dir_patches,
              device,
              ps,
              epochs,
              batch_size,
              learning_rate,
              dir_checkpoint,
              checkpoint_period,
              amp: bool = False):

    transforms = T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.RandomVerticalFlip(p=1.0),
        T.ToTensor()
    ])

    # 1. Create utils
    train_dataset = ImageDataset(img_dir=dir_patches, ps=ps, train=True, transform=transforms)
    valid_dataset = ImageDataset(img_dir=dir_patches, ps=ps, train=False, transform=T.ToTensor())

    # 2. Split into train / validation partitions
    n_train, n_val = len(train_dataset), len(valid_dataset)

    # 3. Create data loaders
    print('Preparing training data ... \n')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # (Initialize logging)
    experiment = wandb.init(project='I-LMSPEC', resume='allow', anonymous='must', reinit=True)
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent="3%", save_checkpoint=dir_checkpoint,
                                  amp=amp, img_scale=ps), allow_val_change=True)  # img_scale=img_scale was included

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {dir_checkpoint}
        Device:          {device.type}
        Images scaling:  {ps}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    g_optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(net_D.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=drop_rate, gamma=0.5)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=drop_rate, gamma=0.5)
    #grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # d_grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    dict_losses_list = []

    discriminator_loss = DiscLoss()
    gen_loss = GeneratorLoss(size=ps)

    # 5. Begin training
    for epoch in range(epochs):

        net.train()
        net_D.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                exp_images = batch['exp_image']
                gt_images = batch['gt_image']

                assert exp_images.shape[1] == net.n_channels, \
                    'Network has been defined with %d input channels, ' \
                    'but loaded exp_images have %d channels. Please check that ' \
                    'the exp_images are loaded correctly.' % (net.n_channels, exp_images.shape[1])

                assert gt_images.shape[1] == net.n_channels, \
                    'Network has been defined with %d input channels, ' \
                    'but loaded gt_images have %d channels. Please check that ' \
                    'the gt_images are loaded correctly.' % (net.n_channels, gt_images.shape[1])

                # We make Gaussian pyramid object with 4 levels
                GP = GaussianPyramid(4)  # for computing Pyramid Loss
                G_pyramid = GP(gt_images)

                for pyramid in G_pyramid:
                    G_pyramid[pyramid] = G_pyramid[pyramid].to(device=device, dtype=torch.float32)

                laplacian_pyr, y_pred = net(exp_images)
                y_pred = [y for y in y_pred.values()]
                #mae_loss = nn.L1Loss(reduction='sum')
                #bcelog_loss = nn.BCEWithLogitsLoss()
                g_pyramid = [t for t in G_pyramid.values()]

                if (epoch+1 >= 1) and (ps == 256):

                    # Adversarial Loss (only for 256 patches
                    #_, y_pred_2 = net(exp_images)
                    y_pred_2 = [Y.detach() for Y in y_pred]
                    disc_fake = net_D(y_pred_2[-1])
                    #disc_fake = net_D(y_pred['subnet_16'].detach())
                    #fake_loss = bcelog_loss(disc_fake, torch.zeros_like(disc_fake))
                    disc_real = net_D(g_pyramid[-1])
                    #real_loss = bcelog_loss(disc_real, torch.ones_like(disc_real))
                    disc_loss = discriminator_loss(disc_fake, disc_real)


                    #DL = DiscriminatorLoss(net_d=net_D, device=device)
                    #real_loss, fake_loss = DL(G_pyramid['level1'], y_pred['subnet_16'])
                    #disc_loss = (fake_loss + real_loss) #/ 2

                    # DISCRIMINATOR TRAINING
                    d_optimizer.zero_grad()
                    disc_loss.backward(retain_graph=True)
                    d_optimizer.step()

                    #disc_adv = net_D(y_pred['subnet_16'])
                    #adv_loss = bcelog_loss(disc_adv, torch.ones_like(disc_adv))
                    #adv_loss = adversarial_loss(net_D, y_pred['subnet_16'], device)
                    y_pred = [y for y in y_pred.values()]
                    rec_loss, pyr_loss, adv_loss, loss_generator = gen_loss(y_pred, g_pyramid, disc_fake, withoutadvloss=False)


                else:

                    disc_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)
                    #real_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)
                    #fake_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)
                    adv_loss = torch.tensor([[0]]).to(device=device, dtype=torch.float32)
                    rec_loss, pyr_loss, loss_generator = gen_loss(y_pred, g_pyramid, withoutadvloss=True)


                # Generator Loss
                # C = (ps**2)* 3
                # W = (ps ** 2) * 12
                # loss_generator = (1/C)*((4 * mae_loss(y_pred['subnet_24_1'],
                #                               F.interpolate(G_pyramid['level4'],
                #                                             (y_pred['subnet_24_1'].shape[2],
                #                                              y_pred['subnet_24_1'].shape[3]))) +
                #                  2 * mae_loss(y_pred['subnet_24_2'],
                #                               F.interpolate(G_pyramid['level3'],
                #                                             (y_pred['subnet_24_2'].shape[2],
                #                                              y_pred['subnet_24_2'].shape[3]))) +
                #                  mae_loss(y_pred['subnet_24_3'], F.interpolate(G_pyramid['level2'],
                #                                             (y_pred['subnet_24_3'].shape[2],
                #                                              y_pred['subnet_24_3'].shape[3]))) +
                #                  mae_loss(y_pred['subnet_16'], F.interpolate(G_pyramid['level1'],
                #                                             (y_pred['subnet_16'].shape[2],
                #                                              y_pred['subnet_16'].shape[3]))))/y_pred['subnet_16'].shape[0] \
                #                  + W*adv_loss)

                #GENERATOR TRAINING
                g_optimizer.zero_grad()
                loss_generator.backward()
                g_optimizer.step()

                pbar.update(exp_images.shape[0])
                global_step += 1

                epoch_loss += loss_generator.item()

                # if global_step % 50 == 0:
                #     train_report = {'epoch': epoch+1, 'step': global_step, 'Generator loss': loss_generator.item(),
                #                     'Discriminator loss': disc_loss.item(), 'Real loss': real_loss.item(),
                #                     'Fake loss': fake_loss.item(), 'lr': g_optimizer.param_groups[0]['lr']}
                #     dict_losses_list.append(train_report)
                if global_step % 50 == 0:
                    train_report = {'epoch': epoch+1, 'step': global_step, 'Generator loss': loss_generator.item(),
                                    'Discriminator loss': disc_loss.item(), 'Pyr loss': pyr_loss.item(),
                                    'Rec loss': rec_loss.item(), 'lr': g_optimizer.param_groups[0]['lr']}
                    dict_losses_list.append(train_report)

                # LOSSES LOGS
                experiment.log({
                    #'train loss': final_loss.item(),
                    'Generator loss (batch)': loss_generator.item(),
                    #'Real loss (batch)': real_loss.item(),
                    #'Fake loss (batch)': fake_loss.item(),
                    'Discriminator loss (batch)': disc_loss.item(),
                    'Adversarial loss (batch)': adv_loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

                """Uncomment for displaying input patches, Laplacian and Gaussian pyramid, in wandb."""
                # INPUT AND PYRAMID LOGS
                # with all_logging_disabled():
                #     experiment.log({
                #         'Input Patch': [wandb.Image(exp_images[0].cuda(), caption='Exposed patch'),
                #                         wandb.Image(gt_images[0].cuda(), caption='GT patch')
                #                         ],
                #         'Laplacian Pyr': [wandb.Image(laplacian_pyr['level4'][0].cuda(), caption='Level 4'),
                #                           wandb.Image(laplacian_pyr['level3'][0].cuda(), caption='Level 3'),
                #                           wandb.Image(laplacian_pyr['level2'][0].cuda(), caption='Level 2'),
                #                           wandb.Image(laplacian_pyr['level1'][0].cuda(), caption='Level 1')
                #                           ],
                #         'Gaussian Pyr': [wandb.Image(G_pyramid['level4'][0].cuda(), caption='Level 4'),
                #                          wandb.Image(G_pyramid['level3'][0].cuda(), caption='Level 3'),
                #                          wandb.Image(G_pyramid['level2'][0].cuda(), caption='Level 2'),
                #                          wandb.Image(G_pyramid['level1'][0].cuda(), caption='Level 1')
                #                          ]
                #     })


                pbar.set_postfix(**{#'train loss': final_loss.item(),
                                    'Generator loss (batch)': loss_generator.item(),
                                    #'Real loss (batch)': real_loss.item(),
                                    #'Fake loss (batch)': fake_loss.item(),
                                    'Discriminator loss (batch)': disc_loss.item(),
                                    'Adversarial loss (batch)': adv_loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(epoch, net, net_D, val_loader, device, ps)
                        # scheduler.step(val_score)
                        # scheduler.step()
                        # d_scheduler.step()
                        # d_scheduler.step(val_score)

                        for batch_val in val_loader:
                            exp_images_val = batch_val['exp_image']
                            gt_images_val = batch_val['gt_image']
                            _, y_pred_val = net(exp_images_val)

                            experiment.log({
                                'Validation round': [wandb.Image(exp_images_val[0].cuda(), caption='Exposed'),
                                                     wandb.Image(gt_images_val[0].cuda(),
                                                                 caption='Ground Truth'),
                                                     wandb.Image(y_pred_val['subnet_16'][0].cuda(),
                                                                 caption='Prediction')]
                                })

                        logging.info('Validation loss: {}'.format(val_score.item()))
                        experiment.log({
                            'Gen learning rate': g_optimizer.param_groups[0]['lr'],
                            'Disc learning rate': d_optimizer.param_groups[0]['lr'],
                            'validation score': val_score,
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        scheduler.step()
        if (epoch+1 >= 15) and (ps == 256):
            d_scheduler.step()

        if dir_checkpoint:
            Path(os.path.join(dir_checkpoint, 'main_net')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(dir_checkpoint, 'disc_net')).mkdir(parents=True, exist_ok=True)

            if epoch + 1 % checkpoint_period == 0:
                torch.save(net.state_dict(),
                           os.path.join(dir_checkpoint, 'main_net', 'chckpnt_epoch{}_{}.pth'.format(epoch + 1, ps)))
                torch.save(net_D.state_dict(),
                           os.path.join(dir_checkpoint, 'disc_net', 'D_chckpnt_epoch{}_{}.pth'.format(epoch + 1, ps)))
                logging.info(f'Checkpoint at {epoch + 1} saved!')

            if epoch == epochs - 1:
                torch.save(net.state_dict(),
                           os.path.join(dir_checkpoint, 'main_net', 'model_{}.pth'.format(ps)))
                torch.save(net_D.state_dict(),
                           os.path.join(dir_checkpoint, 'disc_net', 'D_model_{}.pth'.format(ps)))

                logging.info(f'Last checkpoint at {epoch + 1} saved!')

            # Make a report of the losses
            df = pd.DataFrame(data=dict_losses_list)
            #filepath = Path(os.path.join(dir_checkpoint, 'train_report_P%d.csv'%ps))
            #df.to_csv(filepath)

            experiment.log({'Train report P%d'%ps: wandb.Table(data=df)})

    experiment.finish()
# if __name__ == '__main__':


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)