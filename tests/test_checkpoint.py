import argparse
from unittest import TestCase

import torch
import yaml

from AnimeGANInitTrain import AnimeGANInitTrain
from AnimeGANv2 import AnimeGANv2
from tools.utils import *
from matplotlib import pyplot as plt
import cv2


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

    config_dict = yaml.safe_load(open('../config/config-defaults.yaml', 'r'))
    model = AnimeGANInitTrain(parser.parse_args())
    model.load_from_checkpoint('../checkpoint/animeGan/Hayao/epoch=4-step=3330-v1.ckpt', strict=False)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    savedir = '../checkpoint/animeGan/Hayao/epoch=4-step=3330-v1.ckpt'

    state = torch.load(savedir, map_location='cpu')
    print(state)

class TestSuccessLoadCheckpoint(TestCase):
    config_dict = yaml.safe_load(open('../config/config-defaults.yaml', 'r'))
    model = AnimeGANv2(ch=64, n_dis=3, img_size=[256, 256], **config_dict['model'])
    sample_file = '../dataset/test/HR_photo/1 (1).jpg'
    img = cv2.imread(sample_file).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    sample_image = np.asarray(load_test_data(sample_file))
    sample_image = torch.Tensor(sample_image)
    with torch.no_grad():
        y_hat = model(sample_image)
        test_generated_predict = y_hat.permute(0, 2, 3, 1).cpu().detach().numpy()
        test_generated_predict = np.squeeze(test_generated_predict, axis=0)
        test_generated_predict = (test_generated_predict + 1.) / 2 * 255
        test_generated_predict = np.clip(test_generated_predict, 0, 255).astype(np.uint8)
        test_generated_predict = cv2.cvtColor(test_generated_predict, cv2.COLOR_BGR2RGB)
        plt.imshow(test_generated_predict)
        plt.show()

    checkpoint = torch.load('../checkpoint/animeGan/Hayao/epoch=4-step=3330-v1.ckpt', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    with torch.no_grad():
        y_hat = model(sample_image)
        test_generated_predict = y_hat.permute(0, 2, 3, 1).cpu().detach().numpy()
        test_generated_predict = np.squeeze(test_generated_predict, axis=0)
        test_generated_predict = (test_generated_predict + 1.) / 2 * 255
        test_generated_predict = np.clip(test_generated_predict, 0, 255).astype(np.uint8)
        test_generated_predict = cv2.cvtColor(test_generated_predict, cv2.COLOR_BGR2RGB)
        plt.imshow(test_generated_predict)
        plt.show()

    checkpoint = torch.load('../checkpoint/animeGan/Hayao/epoch=39-step=66560.ckpt', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    with torch.no_grad():
        y_hat = model(sample_image)
        test_generated_predict = y_hat.permute(0, 2, 3, 1).cpu().detach().numpy()
        test_generated_predict = np.squeeze(test_generated_predict, axis=0)
        test_generated_predict = (test_generated_predict + 1.) / 2 * 255
        test_generated_predict = np.clip(test_generated_predict, 0, 255).astype(np.uint8)
        test_generated_predict = cv2.cvtColor(test_generated_predict, cv2.COLOR_BGR2RGB)
        plt.imshow(test_generated_predict)
        plt.show()