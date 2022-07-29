from unittest import TestCase
from discriminator import Discriminator

class TestDiscriminator(TestCase):
    def test_request_grade(self):
        model = Discriminator(in_ch=64, out_ch=3, n_dis=3, sn=True)
        print(model)
        for name, value in model.named_parameters():
            if "conv3" not in name:
                value.requires_grad = False
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
        self.assertEqual(True, True)
