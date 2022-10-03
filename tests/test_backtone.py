import os
import sys
from unittest import TestCase

import numpy as np
import torch
from PIL import Image

from net.backtone import VGGCaffePreTrained
from tools.utils import preprocessing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class TestVGGCaffePreTrained(TestCase):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGGCaffePreTrained().eval()
    model.setup(device=device)

    image = Image.open("../dataset/test/test_photo256/0.png")
    image = image.resize((224, 224))
    np_img = np.array(image).astype('float32')
    np_img = preprocessing(np_img)

    img = torch.from_numpy(np_img)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    feat = model(img)

    print(feat.shape)
