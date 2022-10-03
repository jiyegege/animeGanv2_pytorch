from collections import OrderedDict
from glob import glob

import pytorch_lightning as pl
import torch.nn
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
    def __init__(self, img_size=None, dataset_name=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if img_size is None:
            img_size = [256, 256]
        self.img_size = img_size
        self.p_model = VGGCaffePreTrained().eval()
        """ Define Generator """
        self.generated = Generator()

    def on_fit_start(self):
        self.p_model.setup(self.device)

    def forward(self, img):
        return self.generated(img)

    def training_step(self, batch, batch_idx):
        real, anime, anime_gray, anime_smooth = batch
        generator_images = self.generated(real)
        # init pharse
        init_c_loss = con_loss(self.p_model, real, generator_images)
        init_loss = self.hparams.con_weight * init_c_loss

        self.log('init_loss', init_loss, on_step=True, prog_bar=True, logger=True)
        return init_loss

    def on_fit_end(self) -> None:
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
                        wandb.Image(test_generated_predict, caption="Name:{}, epoch:{}".format(i, self.current_epoch)))
        wandb.log({"val_images": val_images})

    def configure_optimizers(self):
        G_optim = Adam(self.generated.parameters(), lr=self.hparams.init_lr, betas=(0.5, 0.999))
        return G_optim
