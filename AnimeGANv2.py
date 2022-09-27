import argparse
from collections import OrderedDict
from glob import glob

import pytorch_lightning as pl
import torch.nn
import yaml
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
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        config_dict = yaml.safe_load(open(args.config_path, 'r'))
        # Initialize a new wandb run
        wandb.init(project="AnimeGanV2_pytorch", entity="roger_ds", sync_tensorboard=True, config=config_dict)

        self.img_size = args.img_size

        self.p_model = VGGCaffePreTrained()

        """ Define Generator, Discriminator """
        self.generated = Generator()
        self.discriminator = Discriminator(args.ch, 3, args.n_dis, wandb.config.model['sn'])

        print()
        print("##### Information #####")
        print("# dataset : ", wandb.config.dataset['name'])
        print("# batch_size : ", wandb.config.dataset['batch_size'])
        print("# epoch : ", wandb.config.trainer['epoch'])
        print("# training image size [H, W] : ", self.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,sty_weight,color_weight,tv_weight : ",
              wandb.config.model['g_adv_weight'],
              wandb.config.model['d_adv_weight'],
              wandb.config.model['con_weight'],
              wandb.config.model['sty_weight'],
              wandb.config.model['color_weight'],
              wandb.config.model['tv_weight'])
        print("#g_lr,d_lr : ", wandb.config.model['g_lr'], wandb.config.model['d_lr'])
        print(f"# training_rate G -- D: {wandb.config.trainer['training_rate']} : 1")
        print()

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
            loss = wandb.config.model['real_loss_weight'] * real_loss \
                   + wandb.config.model['fake_loss_weight'] * fake_loss \
                   + wandb.config.model['gray_loss_weight'] * gray_loss \
                   + wandb.config.model['real_blur_loss_weight'] * real_blur_loss
            d_loss = wandb.config.model['d_adv_weight'] * loss

            log_dict = {'Discriminator_loss': d_loss, 'Discriminator_real_loss': real_loss,
                        'Discriminator_fake_loss': fake_loss,
                        'Discriminator_gray_loss': gray_loss, 'Discriminator_real_blur_loss': real_blur_loss}
            self.log('D_loss', d_loss, on_epoch=False, on_step=True, prog_bar=True)
            output = OrderedDict({
                'loss': d_loss,
                'log': log_dict
            })
            return output
        # train generator
        elif optimizer_idx == 1:
            c_loss, s_loss = con_sty_loss(self.p_model, real, anime_gray, fake_image)
            tv_loss = wandb.config.model['tv_weight'] * total_variation_loss(fake_image)
            col_loss = color_loss(real, fake_image)
            t_loss = wandb.config.model['con_weight'] * c_loss \
                     + wandb.config.model['sty_weight'] * s_loss \
                     + wandb.config.model['color_weight'] * col_loss \
                     + tv_loss
            g_loss = wandb.config.model['g_adv_weight'] * generator_loss(generated_logit)
            Generator_loss = t_loss + g_loss

            log_dict = {'Generator_loss': Generator_loss, 'Generator_con_loss': c_loss, 'Generator_sty_loss': s_loss,
                        'Generator_color_loss': col_loss}
            self.log('G_loss', Generator_loss, on_epoch=False, on_step=True, prog_bar=True)
            output = OrderedDict({
                'loss': Generator_loss,
                'log': log_dict
            })
            return output

    def training_epoch_end(self, batch_parts):
        # log epoch metrics to wandb
        log_dict = batch_parts[len(batch_parts) - 1]
        for item in log_dict:
            for key, value in item['log'].items():
                wandb.log({key: value})

        # log epoch images to wandb
        val_files = glob('./dataset/{}/*.*'.format('val'))
        val_images = []
        for i, sample_file in enumerate(val_files):
            print('val: ' + str(i) + sample_file)
            self.generated.eval()
            with torch.no_grad():
                sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                test_real = torch.from_numpy(sample_image).type_as(self.generated.model[0].weight)
                test_generated_predict = self.generated(test_real)
                test_generated_predict = test_generated_predict.permute(0, 2, 3, 1).cpu().detach().numpy()
                test_generated_predict = np.squeeze(test_generated_predict, axis=0)
                if i == 0 or i == 26 or i == 5:
                    val_images.append(
                        wandb.Image(test_generated_predict,
                                    caption="Name:{}, epoch:{}".format(i, self.current_epoch + 10)))
        wandb.log({"val_images": val_images})

    def configure_optimizers(self):
        G_optim = Adam(self.generated.parameters(), lr=wandb.config.model['g_lr'], betas=(0.5, 0.999))
        D_optim = Adam(self.discriminator.parameters(), lr=wandb.config.model['d_lr'], betas=(0.5, 0.999))
        return [D_optim, G_optim], []
