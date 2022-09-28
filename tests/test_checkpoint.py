import argparse
from unittest import TestCase

from AnimeGANInitTrain import AnimeGANInitTrain


class TestCheckPoint(TestCase):
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config_path', type=str, help='hyper params config path', default='config/config-defaults.yaml')
    parser.add_argument('--img_size', type=list, default=[256, 256], help='The size of image: H and W')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')
    parser.add_argument('--hyperparameters', type=str, default='False')
    parser.add_argument('--pre_train_weight', type=str, required=False,
                        help='pre-trained weight path, tensorflow checkpoint directory')
    parser.add_argument('--init_train_flag', type=str, default='False')

    model = AnimeGANInitTrain(parser.parse_args())
    model.load_from_checkpoint('../checkpoint/epoch=4-step=4160.ckpt', strict=False)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())