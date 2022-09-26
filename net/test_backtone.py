import os
from unittest import TestCase

import torch
from matplotlib import pyplot as plt

from backtone import VGGCaffePreTrained
from PIL import Image
import numpy as np
from tools.utils import preprocessing

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class TestVGGCaffePreTrained(TestCase):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGGCaffePreTrained()
    model.setup(device=device)

    image = Image.open("../dataset/test/test_photo256/0.png")
    image = image.resize((224, 224))
    np_img = np.array(image).astype('float32')
    np_img = preprocessing(np_img, [224, 224])

    img = torch.from_numpy(np_img)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    feat = model(img)

    print(feat.shape)
