import argparse
import logging
import os
import torch
import torchvision.transforms as T
from PIL import Image
from generator import Generator


def predict_img(net,
                full_img,
                device):
    net.eval()
    transform = T.ToTensor()
    img = transform(full_img)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        trans_PIL = T.ToPILImage()
        output = trans_PIL(output['subnet_16'][0])

    return output


def get_args():
    parser = argparse.ArgumentParser(description='Test model with exposed images')
    parser.add_argument('--model', '-m', default='checkpoint/main_net/model_256.pth', metavar='FILE',
                        help='Specify the path in which the model is stored')
    parser.add_argument('--input_dir', '-i', metavar='INPUT', help='Directory of input images', required=True)
    parser.add_argument('--output_dir', '-o', metavar='OUTPUT', help='Directory of output images')
    parser.add_argument("--GPU", type=int, default=1,
                        help="Select the device")
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_args()
    in_files = os.listdir(opt.input_dir)
    out_files = opt.output_dir

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

    for filename in in_files:

        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(os.path.join(opt.input_dir, filename))

        y_pred = predict_img(net=net,
                             full_img=img,
                             device=device)

        original_name = os.path.split(filename)[1]
        new_path = os.path.join(out_files, original_name)
        y_pred.save(new_path, format="png")
        logging.info(f'Image {original_name} was saved in {os.path.split(new_path)[0]}')
