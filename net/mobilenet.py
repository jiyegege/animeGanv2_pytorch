import torch
import torch.nn as nn
import torchvision.models as models


class Mobilenet(nn.Module):
    def __init__(self):
        super(Mobilenet, self).__init__()
        self.backbone = self.get_backbone().eval()

    def forward(self, x):
        return self.backbone(x)

    @staticmethod
    def get_backbone():
        model = models.mobilenet_v2(pretrained=True)
        #print(model)
        layers = []
        i = 0
        for layer in model.features:
            if i < 6 and isinstance(layer, nn.Module):
                layers.append(layer)
                i += 1
            else:
                if i == 6 and isinstance(layer, nn.Module):
                    for conv in layer.children():
                        if isinstance(conv, nn.Module):
                            for item in conv.children():
                                if isinstance(item, nn.Module):
                                    for l in item.children():
                                        layers.append(l)
                                        break
                                    break
                            break
                    break

        backbone = nn.Sequential(*layers)
        return backbone
