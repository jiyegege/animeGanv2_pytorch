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
    parser.add_argument('--if_adjust_brightness', action='store_true',
                        help='adjust brightness by the real photo')
    parser.add_argument('--test_file_path', type=str, default=None,
                        help='test file path')

    """checking arguments"""

    return parser.parse_args()


def load_model(model_dir):
    """
    load model from checkpoint
    Args:
        model_dir: checkpoint directory

    Returns: model

    """
    ckpt = torch.load(model_dir, map_location=device)
    generated = Generator()
    generatordict = dict(filter(lambda k: 'generated' in k[0], ckpt['state_dict'].items()))
    generatordict = {k.split('.', 1)[1]: v for k, v in generatordict.items()}
    generated.load_state_dict(generatordict, True)
    # model.summary()
    generated.eval()
    del generatordict
    del ckpt
    return generated


def test(model_dir, test_file_path, if_adjust_brightness):
    # tf.reset_default_graph()
    result_dir = 'results'
    check_folder(result_dir)

    generated = load_model(model_dir)
    # print('Processing image: ' + sample_file)

    sample_image = np.asarray(load_test_data(test_file_path))
    sample_image = torch.Tensor(sample_image)
    image_path = os.path.join(result_dir, '{0}'.format(os.path.basename(test_file_path)))
    fake_img = generated(sample_image).detach().numpy()
    fake_img = np.squeeze(fake_img, axis=0)
    fake_img = np.transpose(fake_img, (1, 2, 0))
    if if_adjust_brightness:
        save_images(fake_img, image_path, test_file_path)
    else:
        save_images(fake_img, image_path, None)
    print('Saved image: ' + image_path)


if __name__ == '__main__':
    arg = parse_args()
    print(arg.model_dir)
    test(arg.model_dir, arg.test_file_path, arg.if_adjust_brightness)
