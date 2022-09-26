import argparse
import os

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from AnimeGANv2 import AnimeGANv2
from AnimeGANInitTrain import AnimeGANInitTrain
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
    parser.add_argument('--init_train_flag', type=str, required=True, default='False')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    if args.init_train_flag.lower() == 'true':
        model = AnimeGANInitTrain(args)
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint/initAnimeGan'), monitor='epoch',
                                              mode='max', save_top_k=-1)
    else:
        model = AnimeGANv2(args)
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoint/animeGan', args.dataset), save_top_k=-1,
                                              monitor='epoch', mode='max')
    config_dict = yaml.safe_load(open(args.config_path, 'r'))

    dataModel = AnimeGANDataModel(data_dir=config_dict['dataset']['path'],
                                  dataset=config_dict['dataset']['name'],
                                  batch_size=config_dict['dataset']['batch_size'],
                                  num_workers=config_dict['dataset']['num_workers'])
    if args.pre_train_weight:
        print("Load from checkpoint:", args.pre_train_weight)
        model.load_from_checkpoint(args.pre_train_weight, strict=False)
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=5,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dataModel)
    print(" [*] Training finished!")


if __name__ == '__main__':
    main()
