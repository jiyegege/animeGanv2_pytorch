import unittest

import numpy as np
import onnx
import onnxruntime
import torch

from tools.utils import *
from matplotlib import pyplot as plt
import cv2


class MyTestCase(unittest.TestCase):
    def test_onnx_model(self):
        onnx_model = onnx.load("../save_model/onnx/animeGan.onnx")
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession("../save_model/onnx/animeGan_dynamic.onnx")
        input_name = ort_session.get_inputs()[0].name

        sample_image = np.asarray(load_test_data("../dataset/test/HR_photo/1 (1).jpg"))

        ort_inputs = {input_name: sample_image}

        ort_outs = ort_session.run(None, ort_inputs)

        img_out_y = ort_outs[0]
        test_generated_predict = np.squeeze(img_out_y, axis=0)
        test_generated_predict = (test_generated_predict + 1.) / 2 * 255
        test_generated_predict = np.clip(test_generated_predict, 0, 255).astype(np.uint8)
        test_generated_predict = np.transpose(test_generated_predict, (1, 2, 0))
        test_generated_predict = cv2.cvtColor(test_generated_predict, cv2.COLOR_BGR2RGB)
        plt.imshow(test_generated_predict)
        plt.show()

    def test_pytorch_model(self):
        model = torch.load("../save_model/pytorch/animeGan.pth", map_location='cpu')
        model.eval()

        sample_image = np.asarray(load_test_data("../dataset/test/HR_photo/1 (1).jpg"))
        sample_image = torch.Tensor(sample_image)
        gerated_predict = model(sample_image)
        gerated_predict = gerated_predict.permute(0, 2, 3, 1).cpu().detach().numpy()
        test_generated_predict = np.squeeze(gerated_predict, axis=0)
        test_generated_predict = (test_generated_predict + 1.) / 2 * 255
        test_generated_predict = np.clip(test_generated_predict, 0, 255).astype(np.uint8)
        test_generated_predict = cv2.cvtColor(test_generated_predict, cv2.COLOR_BGR2RGB)
        plt.imshow(test_generated_predict)
        plt.show()


if __name__ == '__main__':
    unittest.main()
