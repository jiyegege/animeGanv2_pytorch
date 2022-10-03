import argparse

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from AnimeGANInitTrain import AnimeGANInitTrain
from AnimeGANv2 import AnimeGANv2
from tools.AnimeGanDataModel import AnimeGANDataModel
from tools.utils import *

"""parsing and configuration"""


def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config_path', type=str, help='hyper params config path', required=True)
    parser.add_argument('--img_size', type=list, default=[256, 256], help='The size of image: H and W')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')
    parser.add_argument('--hyperparameters', type=str, default='False')
    parser.add_argument('--pre_train_weight', type=str, required=False,
                        help='pre-trained weight path, tensorflow checkpoint directory')
    parser.add_argument('--resume_ckpt_path', type=str, required=False, help='resume checkpoint path')
    parser.add_argument('--init_train_flag', type=str, required=True, default='False')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --epoch
    try:
        assert args.config_path
    except:
        print('config_path is required')
    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    config_dict = yaml.safe_load(open(args.config_path, 'r'))
    if args.init_train_flag.lower() == 'true':
        model = AnimeGANInitTrain(args.img_size, config_dict['dataset']['name'], **config_dict['model'])
        check_folder(os.path.join('checkpoint/initAnimeGan', config_dict['dataset']['name']))
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint/initAnimeGan', config_dict['dataset']['name']),
                                              monitor='epoch',
                                              mode='max',
                                              save_top_k=-1)
        tensorboard_logger = TensorBoardLogger(save_dir='logs/initAnimeGan')
        wandb_logger = WandbLogger(project='AnimeGanV2_init_pytorch', name='initAnimeGan')
        trainer = Trainer(
            accelerator='auto',
            max_epochs=config_dict['trainer']['epoch'],
            callbacks=[checkpoint_callback],
            logger=[tensorboard_logger, wandb_logger],
            precision=16
        )
        print()
        print("##### Information #####")
        print("# dataset : ", config_dict['dataset']['name'])
        print("# batch_size : ", config_dict['dataset']['batch_size'])
        print("# epoch : ", config_dict['trainer']['epoch'])
        print("# training image size [H, W] : ", args.img_size)
        print("#con_weight,sty_weight : ", config_dict['model']['con_weight'])
        print("#init_lr: ", config_dict['model']['init_lr'])
        print()
    else:
        model = AnimeGANv2(args.ch, args.n_dis, args.img_size, config_dict['dataset']['name'], args.pre_train_weight,
                           **config_dict['model'])
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint/animeGan', config_dict['dataset']['name']),
                                              save_top_k=-1,
                                              monitor='epoch', mode='max')
        tensorboard_logger = TensorBoardLogger(save_dir='logs/animeGan')
        wandb_logger = WandbLogger(project='AnimeGanV2_pytorch', name='animeGan')
        trainer = Trainer(
            accelerator='auto',
            max_epochs=config_dict['trainer']['epoch'],
            callbacks=[checkpoint_callback],
            logger=[tensorboard_logger, wandb_logger],
            precision=16
        )
        print()
        print("##### Information #####")
        print("# dataset : ", config_dict['dataset']['name'])
        print("# batch_size : ", config_dict['dataset']['batch_size'])
        print("# epoch : ", config_dict['trainer']['epoch'])
        print("# training image size [H, W] : ", args.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,sty_weight,color_weight,tv_weight : ",
              config_dict['model']['g_adv_weight'],
              config_dict['model']['d_adv_weight'],
              config_dict['model']['con_weight'],
              config_dict['model']['sty_weight'],
              config_dict['model']['color_weight'],
              config_dict['model']['tv_weight'])
        print("#g_lr,d_lr : ", config_dict['model']['g_lr'], config_dict['model']['d_lr'])
        print()

    dataModel = AnimeGANDataModel(data_dir=config_dict['dataset']['path'],
                                  dataset=config_dict['dataset']['name'],
                                  batch_size=config_dict['dataset']['batch_size'],
                                  num_workers=config_dict['dataset']['num_workers'])
    if args.ckpt_path:
        print("resume from checkpoint:", args.ckpt_path)
        trainer.fit(model, dataModel, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, dataModel)
    model.to_onnx('animeGan.onnx', input_sample=torch.randn(1, 3, 256, 256))
    torch.save(model.generated.state_dict(), 'animeGan.pth')
    print(" [*] Training finished!")


if __name__ == '__main__':
    main()
