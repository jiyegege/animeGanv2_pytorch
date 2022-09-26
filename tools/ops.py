import torch
from torch import nn


def generator_loss(fake):
    fake_loss = torch.mean(torch.square(fake - 1.0))
    return fake_loss


def discriminator_loss(real, gray, fake, real_blur):
    real_loss = torch.mean(torch.square(real - 1.0))
    gray_loss = torch.mean(torch.square(gray))
    fake_loss = torch.mean(torch.square(fake))
    real_blur_loss = torch.mean(torch.square(real_blur))

    # for Hayao : 1.2, 1.2, 1.2, 0.8
    # for Paprika : 1.0, 1.0, 1.0, 0.005
    # for Shinkai: 1.7, 1.7, 1.7, 1.0
    return real_loss, fake_loss, gray_loss, real_blur_loss


def gram_matrix(input):
    b, c, h, w = input.size()
    reshape_input = input.view(b * c, h * w)
    G = torch.mm(reshape_input, reshape_input.t())
    return G.div(b * c * h * w)


def con_loss(pre_train_model: nn.Module, real, fake):
    real_feature_map = pre_train_model(real)
    fake_feature_map = pre_train_model(fake)
    loss = nn.L1Loss()(real_feature_map, fake_feature_map)
    return loss


def style_loss(style, fake):
    return nn.L1Loss()(gram_matrix(style), gram_matrix(fake))


def con_sty_loss(pre_train_model: nn.Module, real, anime, fake):
    real_feature_map = pre_train_model(real)
    fake_feature_map = pre_train_model(fake)
    anime_feature_map = pre_train_model(anime)

    c_loss = nn.L1Loss()(real_feature_map, fake_feature_map)
    s_loss = style_loss(anime_feature_map, fake_feature_map)

    return c_loss, s_loss


def color_loss(real, fake):
    real_yuv = rgb2yuv(real)
    fake_yuv = rgb2yuv(fake)
    loss = nn.L1Loss()(real_yuv[:, :, :, 0], fake_yuv[:, :, :, 0]) + \
           nn.SmoothL1Loss()(real_yuv[:, :, :, 1], fake_yuv[:, :, :, 1]) + \
           nn.SmoothL1Loss()(real_yuv[:, :, :, 2], fake_yuv[:, :, :, 2])
    return loss


def total_variation_loss(inputs):
    dh = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dw = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    return torch.mean(torch.abs(dh)) + torch.mean(torch.abs(dw))


def rgb2yuv(rgb):
    rgb_ = (rgb + 1.0) / 2.0
    # from  Wikipedia
    A = torch.tensor([[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]])
    A = A.type_as(rgb)
    yuv = torch.tensordot(rgb_, A, dims=([rgb.ndim - 3], [0]))
    return yuv
