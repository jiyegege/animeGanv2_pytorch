from unittest import TestCase

import cv2
import torch
import yaml
from matplotlib import pyplot as plt

from AnimeGANv2 import AnimeGANv2
from tools.utils import *


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

    checkpoint = torch.load('../checkpoint/animeGan/Hayao/epoch=59-step=79920-v1.ckpt', map_location='cpu')
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