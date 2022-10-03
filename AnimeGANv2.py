from collections import OrderedDict
from glob import glob

import pytorch_lightning as pl
import torch.nn
from torch.optim import Adam

import wandb
from net.backtone import VGGCaffePreTrained
from net.discriminator import Discriminator
from net.generator import Generator
from tools.ops import *
from tools.utils import *


##################################################################################
# Model
##################################################################################
class AnimeGANv2(pl.LightningModule):
    def __init__(self, ch=64, n_dis=3, img_size=None, dataset_name=None, pre_trained_ckpt: str = None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if img_size is None:
            img_size = [256, 256]
        self.img_size = img_size
        self.p_model = VGGCaffePreTrained().eval()
        """ Define Generator, Discriminator """
        self.generated = Generator()
        self.discriminator = Discriminator(ch, 3, n_dis, kwargs['sn'])
        self.pre_trained_ckpt = pre_trained_ckpt

    def setup(self, stage) -> None:
        if stage == 'fit':
            if self.pre_trained_ckpt is not None:
                ckpt = torch.load(self.pre_trained_ckpt, map_location=self.device)
                generatordict = dict(filter(lambda k: 'generated' in k[0], ckpt['state_dict'].items()))
                generatordict = {k.split('.', 1)[1]: v for k, v in generatordict.items()}
                self.generated.load_state_dict(generatordict, True)
                print('Load pre-trained generator from {}'.format(self.pre_trained_ckpt))
                del generatordict
                del ckpt
        elif stage == 'test':
            pass

    def on_fit_start(self):
        self.p_model.setup(self.device)

    def forward(self, img):
        return self.generated(img)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, anime, anime_gray, anime_smooth = batch
        fake_image = self.generated(real)
        generated_logit = self.discriminator(fake_image)
        # train discriminator
        if optimizer_idx == 0:
            d_anime_logit = self.discriminator(anime)
            d_anime_gray_logit = self.discriminator(anime_gray)
            d_smooth_logit = self.discriminator(anime_smooth)

            """ Define Loss """
            (real_loss, fake_loss, gray_loss, real_blur_loss) = discriminator_loss(d_anime_logit, d_anime_gray_logit,
                                                                                   generated_logit,
                                                                                   d_smooth_logit)
            loss = self.hparams.real_loss_weight * real_loss \
                   + self.hparams.fake_loss_weight * fake_loss \
                   + self.hparams.gray_loss_weight * gray_loss \
                   + self.hparams.real_blur_loss_weight * real_blur_loss
            d_loss = self.hparams.d_adv_weight * loss

            self.log('D_loss', d_loss, on_epoch=False, on_step=True, prog_bar=True, logger=True)
            self.log('D_real_loss', real_loss, on_epoch=True, on_step=True, prog_bar=False, logger=True)
            self.log('D_fake_loss', fake_loss, on_epoch=True, on_step=True, prog_bar=False, logger=True)
            self.log('D_gray_loss', gray_loss, on_epoch=True, on_step=True, prog_bar=False, logger=True)
            self.log('D_real_blur_loss', real_blur_loss, on_epoch=True, on_step=True, prog_bar=False, logger=True)
            return d_loss
        # train generator
        elif optimizer_idx == 1:
            c_loss, s_loss = con_sty_loss(self.p_model, real, anime_gray, fake_image)
            tv_loss = self.hparams.tv_weight * total_variation_loss(fake_image)
            col_loss = color_loss(real, fake_image)
            t_loss = self.hparams.con_weight * c_loss \
                     + self.hparams.sty_weight * s_loss \
                     + self.hparams.color_weight * col_loss \
                     + tv_loss
            g_loss = self.hparams.g_adv_weight * generator_loss(generated_logit)
            Generator_loss = t_loss + g_loss

            self.log('G_loss', Generator_loss, on_epoch=False, on_step=True, prog_bar=True)
            self.log('G_con_loss', c_loss, on_epoch=True, on_step=True, prog_bar=False, logger=True)
            self.log('G_sty_loss', s_loss, on_epoch=True, on_step=True, prog_bar=False, logger=True)
            self.log('G_color_loss', col_loss, on_epoch=True, on_step=True, prog_bar=False, logger=True)
            return Generator_loss

    def training_epoch_end(self, batch_parts):
        # log epoch images to wandb
        val_files = glob('./dataset/{}/*.*'.format('val'))
        val_images = []
        for i, sample_file in enumerate(val_files):
            print('val: ' + str(i) + sample_file)
            self.generated.eval()
            if i == 0 or i == 26 or i == 5:
                with torch.no_grad():
                    sample_image = np.asarray(load_test_data(sample_file))
                    test_real = torch.from_numpy(sample_image).type_as(self.generated.out_layer[0].weight)
                    test_generated_predict = self.generated(test_real)
                    test_generated_predict = test_generated_predict.permute(0, 2, 3, 1).cpu().detach().numpy()
                    test_generated_predict = np.squeeze(test_generated_predict, axis=0)
                    val_images.append(
                        wandb.Image(test_generated_predict,
                                    caption="Name:{}, epoch:{}".format(i, self.current_epoch + 10)))
        wandb.log({"val_images": val_images})

    def configure_optimizers(self):
        G_optim = Adam(self.generated.parameters(), lr=self.hparams.g_lr, betas=(0.5, 0.999))
        D_optim = Adam(self.discriminator.parameters(), lr=self.hparams.d_lr, betas=(0.5, 0.999))
        return [D_optim, G_optim], []
