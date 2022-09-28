import unittest

import numpy as np
import torch
from PIL import Image

from net.mobilenet import Mobilenet


class MyTestCase(unittest.TestCase):
    def test_something(self):
        model = Mobilenet()
        image = Image.open("../dataset/test/test_photo256/0.png")
        np_img = np.array(image).astype('float32')
        np_img = np.transpose(np_img, (2, 0, 1))
        np_img = np.expand_dims(np_img, axis=0)
        img = torch.from_numpy(np_img)
        features = model(img)
        print(features.shape)
        print(features)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
