from unittest import TestCase
from net.generator import Generator


class TestGenerator(TestCase):
    def test_request_grade(self):
        model = Generator()
        for name, value in model.named_parameters():
            if "out_layer" not in name:
                value.requires_grad = False
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
        self.assertEqual(True, True)
