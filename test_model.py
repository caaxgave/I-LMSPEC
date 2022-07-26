import argparse
import logging
import os
import torch
import torchvision.transforms as T
from PIL import Image
from generator import Generator
from contextlib import contextmanager
import wandb


def predict_img(net,
                full_img,
                device):
    net.eval()
    transform = T.ToTensor()
    img = transform(full_img)
    img = img.to(device=device, dtype=torch.float32)

    outputs = []

    with torch.no_grad():
        lp_pyr, output = net(img)
        trans_PIL = T.ToPILImage()
        outputs.append(trans_PIL(output['subnet_24_1'][0]))
        outputs.append(trans_PIL(output['subnet_24_2'][0]))
        outputs.append(trans_PIL(output['subnet_24_3'][0]))
        outputs.append(trans_PIL(output['subnet_16'][0]))

    return lp_pyr, output


def get_args():
    parser = argparse.ArgumentParser(description='Test model with exposed images')
    parser.add_argument('--model', '-m', default='checkpoint/main_net/model_256.pth', metavar='FILE',
                        help='Specify the path in which the model is stored')
    parser.add_argument('--input_dir', '-i', metavar='INPUT', help='Directory of input images', required=True)
    parser.add_argument('--output_dir', '-o', metavar='OUTPUT', help='Directory of output images')
    parser.add_argument("--GPU", type=int, default=1,
                        help="Select the device")
    return parser.parse_args()


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


if __name__ == '__main__':
    opt = get_args()
    in_files = opt.input_dir
    out_files = opt.output_dir

    exp_images = os.path.join(in_files, 'Underexposed')
    gt_images = os.path.join(in_files, 'Normal_frames')

    # Specify device either GPU or CPU
    if torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device(opt.GPU)
    else:
        device = torch.device('cpu')

    net = Generator(n_channels=3, device=device, bilinear=False)

    logging.info(f'Loading model {opt.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(opt.model, map_location=device))

    logging.info('Model loaded!')

    # (Initialize logging)
    experiment = wandb.init(project='I-LMSPEC_TEST', resume='allow', anonymous='must', reinit=True)
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent="3%", save_checkpoint=dir_checkpoint,
    #                               amp=amp, img_scale=ps), allow_val_change=True)  # img_scale=img_scale was included

    for filename in os.listdir(exp_images):

        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(os.path.join(exp_images, filename))
        gt = Image.open(os.path.join(gt_images, filename))

        lp_pyr, y_pred = predict_img(net=net,
                                     full_img=img,
                                     device=device)

        # INPUT AND PYRAMID LOGS
        with all_logging_disabled():
            experiment.log({
                'Input Patch': [wandb.Image(img, caption='Exposed patch'),
                                wandb.Image(gt, caption='GT patch')
                                ],
                'Laplacian Pyr': [wandb.Image(lp_pyr['level4'], caption='Level 4'),
                                  wandb.Image(lp_pyr['level3'], caption='Level 3'),
                                  wandb.Image(lp_pyr['level2'], caption='Level 2'),
                                  wandb.Image(lp_pyr['level1'], caption='Level 1')
                                  ],
                'Predictions': [wandb.Image(ypred['subnet_24_1'], caption='subnet_24_1'),
                                wandb.Image(ypred['subnet_24_2'], caption='subnet_24_2'),
                                wandb.Image(ypred['subnet_24_3'], caption='subnet_24_3'),
                                wandb.Image(ypred['subnet_16'], caption='subnet_16')]
            })

        experiment.finish()

        #original_name = os.path.split(filename)[1]
        #new_path = os.path.join(out_files, original_name)
        #y_pred.save(new_path, format="png")
        #logging.info(f'Image {original_name} was saved in {os.path.split(new_path)[0]}')
