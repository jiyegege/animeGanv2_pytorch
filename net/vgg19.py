import torch
import torch.nn as nn
import torchvision.models as models

vgg_mean = torch.tensor([0.485, 0.456, 0.406]).float()
vgg_std = torch.tensor([0.229, 0.224, 0.225]).float()

if torch.cuda.is_available():
    vgg_std = vgg_std.cuda()
    vgg_mean = vgg_mean.cuda()


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.vgg19 = self.get_vgg19().eval()
        self.mean = vgg_mean.view(-1, 1, 1)
        self.std = vgg_std.view(-1, 1, 1)

    def forward(self, x):
        return self.vgg19(self.normalize_vgg(x))

    @staticmethod
    def get_vgg19(last_layer='conv4_4'):
        vgg = models.vgg19(pretrained=torch.cuda.is_available()).features
        model_list = []

        i = 0
        j = 1
        for layer in vgg.children():
            if isinstance(layer, nn.MaxPool2d):
                i = 0
                j += 1

            elif isinstance(layer, nn.Conv2d):
                i += 1

            name = f'conv{j}_{i}'

            if name == last_layer:
                model_list.append(layer)
                break

            model_list.append(layer)

        model = nn.Sequential(*model_list)
        return model

    def normalize_vgg(self, image):
        """
        Expect input in range -1 1
        """
        image = (image + 1.0) / 2.0
        return (image - self.mean) / self.std


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    from tools.utils import preprocessing

    image = Image.open("../dataset/test/test_photo256/0.png")
    image = image.resize((224, 224))
    np_img = np.array(image).astype('float32')
    np_img = preprocessing(np_img)

    img = torch.from_numpy(np_img)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)

    vgg = Vgg19()
    print(vgg)

    feat = vgg(img)

    print(feat.shape)
