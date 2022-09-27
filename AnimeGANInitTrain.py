import argparse
from collections import OrderedDict
from glob import glob

import pytorch_lightning as pl
import torch.nn
import yaml
from torch.optim import Adam

import wandb
from net.backtone import VGGCaffePreTrained
from net.generator import Generator
from tools.ops import *
from tools.utils import *


##################################################################################
# Model
##################################################################################
class AnimeGANInitTrain(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        config_dict = yaml.safe_load(open(args.config_path, 'r'))
        # Initialize a new wandb run
        wandb.init(project="AnimeGanV2_init_train_pytorch", entity="roger_ds", sync_tensorboard=True,
                   config=config_dict)
        self.img_size = args.img_size
        self.p_model = VGGCaffePreTrained()
        """ Define Generator """
        self.generated = Generator()

        print()
        print("##### Information #####")
        print("# dataset : ", wandb.config.dataset['name'])
        print("# batch_size : ", wandb.config.dataset['batch_size'])
        print("# epoch : ", wandb.config.trainer['init_epoch'])
        print("# training image size [H, W] : ", self.img_size)
        print("#con_weight,sty_weight : ", wandb.config.model['con_weight'])
        print("#init_lr: ", wandb.config.model['init_lr'])
        print()

    def on_fit_start(self):
        self.p_model.setup(self.device)

    def forward(self, img):
        return self.generated(img)

    def training_step(self, batch, batch_idx):
        real, anime, anime_gray, anime_smooth = batch
        generator_images = self.generated(real)
        # init pharse
        init_c_loss = con_loss(self.p_model, real, generator_images)
        init_loss = wandb.config.model['con_weight'] * init_c_loss

        log_dict = {'init_loss': init_loss.item()}
        output = OrderedDict({
            'loss': init_loss,
            'log': log_dict
        })
        return output

    def on_fit_end(self) -> None:
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
                        wandb.Image(test_generated_predict, caption="Name:{}, epoch:{}".format(i, self.current_epoch)))
        wandb.log({"val_images": val_images})

    def training_epoch_end(self, batch_parts):
        # log epoch metrics to wandb
        log_dict = batch_parts[len(batch_parts) - 1]
        for item in log_dict:
            for key, value in item['log'].items():
                wandb.log({key: value})

    def configure_optimizers(self):
        G_optim = Adam(self.generated.parameters(), lr=wandb.config.model['init_lr'], betas=(0.5, 0.999))
        return G_optim
