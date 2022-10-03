import torch
import argparse
from tools.utils import *
import os
from tqdm import tqdm
from glob import glob
import time
import numpy as np
from net.generator import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_dir', type=str, default='save_model/' + 'generated_Hayao.pth',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--test_dir', type=str, default='dataset/test/t',
                        help='Directory name of test photos')
    parser.add_argument('--save_dir', type=str, default='Shinkai/t',
                        help='what style you want to get')
    parser.add_argument('--if_adjust_brightness', type=bool, default=True,
                        help='adjust brightness by the real photo')

    """checking arguments"""

    return parser.parse_args()


def load_model(model_dir):
    model = torch.load(model_dir, map_location=device)
    # model.summary()
    return model


def test(model_dir, style_name, test_dir, if_adjust_brightness, img_size=[256, 256]):
    # tf.reset_default_graph()
    result_dir = 'results/' + style_name
    check_folder(result_dir)

    test_generated = load_model(model_dir)

    # stats_graph(tf.get_default_graph())

    # print('Processing image: ' + sample_file)
    sample_file = 'dataset/test/test_photo256/31.png'
    sample_image = np.asarray(load_test_data(sample_file))
    sample_image = torch.Tensor(sample_image)
    image_path = os.path.join(result_dir, '{0}'.format(os.path.basename(sample_file)))
    fake_img = test_generated(sample_image).detach().numpy()
    fake_img = np.squeeze(fake_img, axis=0)
    fake_img = np.transpose(fake_img, (1, 2, 0))
    if if_adjust_brightness:
        save_images(fake_img, image_path, sample_file)
    else:
        save_images(fake_img, image_path, None)


if __name__ == '__main__':
    arg = parse_args()
    print(arg.model_dir)
    test(arg.model_dir, arg.save_dir, arg.test_dir, arg.if_adjust_brightness)
