from glob import glob
from multiprocessing import cpu_count

import wandb
import yaml
from tqdm import tqdm

from net.discriminator import Discriminator
from net.generator import Generator
from tools.dataset import AnimeDataSet
from tools.ops import *
from tools.utils import *
from net.vgg19 import Vgg19
from net.mobilenet import Mobilenet

import torch
import torch.nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class AnimeGANv2(object):
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.hyperparameters.lower() == 'true':
            self.hyperparameters = True
        else:
            self.hyperparameters = False

        config_dict = yaml.safe_load(open(args.config_path, 'r'))
        # Initialize a new wandb run
        wandb.init(project="AnimeGanV2_pytorch", entity="roger_ds", sync_tensorboard=True, config=config_dict)

        self.model_name = 'AnimeGANv2'
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.save_freq = args.save_freq

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        """ Discriminator """
        self.n_dis = args.n_dis
        self.ch = args.ch

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        imageDataSet = AnimeDataSet(data_dir='./dataset', dataset=self.dataset_name)
        self.data_loader = DataLoader(imageDataSet,
                                      batch_size=wandb.config.batch_size,
                                      pin_memory=True)
        self.dataset_num = imageDataSet.__len__()
        self.p_model = Vgg19().to(self.device).eval()

        self.pre_train_weight = args.pre_train_weight

        print()
        print("##### Information #####")
        print("# gan type : ", wandb.config.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", wandb.config.batch_size)
        print("# epoch : ", wandb.config.epoch)
        print("# init_epoch : ", wandb.config.init_epoch)
        print("# training image size [H, W] : ", self.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,sty_weight,color_weight,tv_weight : ", wandb.config.g_adv_weight,
              wandb.config.d_adv_weight, wandb.config.con_weight, wandb.config.sty_weight, wandb.config.color_weight,
              wandb.config.tv_weight)
        print("# init_lr,g_lr,d_lr : ", wandb.config.init_lr, wandb.config.g_lr, wandb.config.d_lr)
        print(f"# training_rate G -- D: {wandb.config.training_rate} : 1")
        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self):
        G = Generator()
        return G

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self):
        D = Discriminator(self.ch, 3, self.n_dis, wandb.config.sn)
        return D

    ##################################################################################
    # Model
    ##################################################################################
    def gradient_panalty(self, real, fake, discriminator):
        if wandb.config.gan_type.__contains__('dragan'):
            eps = torch.empty(real.size()).uniform_(0., 1.)
            x_std = torch.std(real)

            fake = real + 0.5 * x_std * eps
        alpha = torch.empty([wandb.config.batch_size, 1, 1, 1]).uniform_(0., 1.)
        interpolated = real + alpha * (fake - real)
        logit = discriminator(interpolated)
        # gradient of D(interpolated)
        grad = torch.autograd.grad(logit, interpolated)[0]
        grad_norm = F.normalize(grad, p=2, dim=1)  # l2 norm

        GP = 0
        # WGAN - LP
        if wandb.config.gan_type.__contains__('lp'):
            GP = wandb.config.ld * torch.mean(torch.square(torch.maximum(0.0, grad_norm - 1.)))

        elif wandb.config.gan_type.__contains__('gp') or wandb.config.gan_type == 'dragan':
            GP = wandb.config.ld * torch.mean(torch.square(grad_norm - 1.))

        return GP

    def train(self):
        """ Define Generator, Discriminator """
        generated = self.generator().to(self.device)
        discriminator = self.discriminator().to(self.device)

        # summary writer
        self.writer = SummaryWriter(self.log_dir + '/' + self.model_dir)

        """ Training """

        init_optim = Adam(generated.parameters(), lr=wandb.config.init_lr, betas=(0.5, 0.999))
        G_optim = Adam(generated.parameters(), lr=wandb.config.g_lr, betas=(0.5, 0.999))
        D_optim = Adam(discriminator.parameters(), lr=wandb.config.d_lr, betas=(0.5, 0.999))

        # saver to save model
        checkpoint = {'generated': generated.state_dict(),
                      'discriminator': discriminator.state_dict(),
                      'p_model': self.p_model.state_dict(),
                      'G_optim': G_optim.state_dict(),
                      'D_optim': D_optim.state_dict()
                      }

        # restore check-point if it exits
        if self.pre_train_weight:
            state = self.load_pre_weight(self.pre_train_weight)
        else:
            state = self.load(self.checkpoint_dir)
        if state:
            if self.pre_train_weight:
                start_epoch = 0
            else:
                start_epoch = state['epoch']
            generated.load_state_dict(state['generated'])
            discriminator.load_state_dict(state['discriminator'])

            for name, value in generated.named_parameters():
                if "out_layer" not in name:
                    value.requires_grad = False

            for name, value in discriminator.named_parameters():
                if "conv3" not in name:
                    value.requires_grad = False

            self.p_model.load_state_dict(state['p_model'])
            G_optim.load_state_dict(state['G_optim'])
            D_optim.load_state_dict(state['D_optim'])
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            print(" [!] Load failed...")

        init_mean_loss = []
        mean_loss = []
        j = wandb.config.training_rate
        for epoch in range(start_epoch, wandb.config.epoch):
            total_step = len(self.data_loader)
            with tqdm(range(total_step)) as tbar:
                for step, data in enumerate(self.data_loader, 0):
                    real = data[0].to(self.device)
                    generated.train()
                    if epoch < wandb.config.init_epoch:
                        init_loss = self.init_step(epoch, generated, init_mean_loss, init_optim, real)
                        tbar.set_description('Epoch %d' % epoch)
                        tbar.set_postfix(init_v_loss=init_loss.item(), mean_v_loss=np.mean(init_mean_loss))
                        tbar.update()
                        if (step + 1) % 200 == 0:
                            init_mean_loss.clear()
                    else:
                        anime = data[1].to(self.device)
                        anime_gray = data[2].to(self.device)
                        anime_smooth = data[3].to(self.device)

                        g_loss, d_loss = self.train_step(anime, anime_smooth, anime_gray, real, G_optim, D_optim,
                                                         discriminator, generated, epoch, j)

                        mean_loss.append([d_loss, g_loss])
                        tbar.set_description('Epoch %d' % epoch)
                        if j == wandb.config.training_rate:
                            tbar.set_postfix(d_loss=d_loss, g_loss=g_loss,
                                             mean_d_loss=np.mean(mean_loss, axis=0)[0],
                                             mean_g_loss=np.mean(mean_loss, axis=0)[1])
                        else:
                            tbar.set_postfix(g_loss=g_loss, mean_g_loss=np.mean(mean_loss, axis=0)[1])
                        tbar.update()

                        if (step + 1) % 200 == 0:
                            mean_loss.clear()

                        j = j - 1
                        if j < 1:
                            j = wandb.config.training_rate
            if not self.hyperparameters:
                if (epoch + 1) >= wandb.config.init_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                    checkpoint['epoch'] = epoch
                    self.save(self.checkpoint_dir, checkpoint)

            if epoch >= wandb.config.init_epoch - 1:
                """ Result Image """
                val_files = glob('./dataset/{}/*.*'.format('val'))
                # save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
                # check_folder(save_path)
                val_images = []
                for i, sample_file in enumerate(val_files):
                    print('val: ' + str(i) + sample_file)
                    generated.eval()
                    with torch.no_grad():
                        sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                        test_real = torch.from_numpy(sample_image).to(self.device)
                        test_generated_predict = generated(test_real).cpu().numpy()
                        test_generated_predict = np.transpose(test_generated_predict, (0, 2, 3, 1))
                        test_generated_predict = np.squeeze(test_generated_predict, axis=0)
                    # save_images(test_real, save_path + '{:03d}_a.jpg'.format(i), None)
                    # save_images(test_generated_predict, save_path + '{:03d}_b.jpg'.format(i), None)
                    if i == 0 or i == 26 or i == 5:
                        val_images.append(
                            wandb.Image(test_generated_predict, caption="Name:{}, epoch:{}".format(i, epoch)))
                        # self.writer.add_image('val_data_' + str(i), test_generated_predict, epoch)
                wandb.log({'val_data': val_images}, step=epoch)
        if not self.hyperparameters:
            save_model_path = 'save_model'
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            torch.save(generated, os.path.join(save_model_path, 'generated_' + self.dataset_name + '.pth'))

    def train_step(self, anime, anime_smooth, anime_gray, real, G_optim,
                   D_optim, discriminator, generated, epoch, j):
        G_optim.zero_grad()
        fake_image = generated(real)
        generated_logit = discriminator(fake_image)

        # gan
        c_loss, s_loss = con_sty_loss(self.p_model, real, anime_gray, fake_image)
        tv_loss = wandb.config.tv_weight * total_variation_loss(fake_image)
        col_loss = color_loss(real, fake_image)
        t_loss = wandb.config.con_weight * c_loss + wandb.config.sty_weight * s_loss + col_loss * wandb.config.color_weight + tv_loss
        g_loss = wandb.config.g_adv_weight * generator_loss(wandb.config.gan_type, generated_logit)
        Generator_loss = t_loss + g_loss
        Generator_loss.backward(retain_graph=True)

        # discriminator
        if j == wandb.config.training_rate:
            D_optim.zero_grad()
            d_anime_logit = discriminator(anime)
            d_anime_gray_logit = discriminator(anime_gray)
            d_smooth_logit = discriminator(anime_smooth)

            """ Define Loss """
            if wandb.config.gan_type.__contains__('gp') or wandb.config.gan_type.__contains__('lp') or \
                    wandb.config.gan_type.__contains__('dragan'):
                GP = self.gradient_panalty(real=anime, fake=fake_image, discriminator=discriminator)
            else:
                GP = 0.0
            d_loss = wandb.config.d_adv_weight * discriminator_loss(wandb.config.gan_type,
                                                                    d_anime_logit,
                                                                    d_anime_gray_logit,
                                                                    generated_logit,
                                                                    d_smooth_logit, epoch, self.writer,
                                                                    wandb.config.real_loss_weight,
                                                                    wandb.config.fake_loss_weight,
                                                                    wandb.config.gray_loss_weight,
                                                                    wandb.config.real_blur_loss_weight) + GP
            d_loss.backward()
            D_optim.step()
        G_optim.step()
        self.writer.add_scalar("Generator_loss", Generator_loss.item(), epoch)
        self.writer.add_scalar("G_con_loss", c_loss.item(), epoch)
        self.writer.add_scalar("G_sty_loss", s_loss.item(), epoch)
        self.writer.add_scalar("G_color_loss", col_loss.item(), epoch)
        self.writer.add_scalar("G_gan_loss", g_loss.item(), epoch)
        self.writer.add_scalar("G_pre_model_loss", t_loss.item(), epoch)

        if j == wandb.config.training_rate:
            self.writer.add_scalar("Discriminator_loss", d_loss.item(), epoch)
        return Generator_loss.item(), d_loss.item()

    def init_step(self, epoch, generated, init_mean_loss, init_optim, real):
        init_optim.zero_grad()
        generator_images = generated(real)
        # init pharse
        init_c_loss = con_loss(self.p_model, real, generator_images)
        init_loss = wandb.config.con_weight * init_c_loss
        init_loss.backward()
        init_optim.step()
        # wandb.log("G_init", init_loss.numpy(), step=epoch)
        self.writer.add_scalar('G_init_loss', init_loss.item(), epoch)
        init_mean_loss.append(init_loss.item())
        return init_loss

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                                   wandb.config.gan_type,
                                                   int(wandb.config.g_adv_weight), int(wandb.config.d_adv_weight),
                                                   int(wandb.config.con_weight), int(wandb.config.sty_weight),
                                                   int(wandb.config.color_weight), int(wandb.config.tv_weight))

    def save(self, checkpoint_dir, state):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, "checkpoint.pth"))

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, "checkpoint.pth")
        if not os.path.exists(checkpoint_dir):
            print(" [!] No checkpoint found...")
            return
        state = torch.load(checkpoint_dir)
        return state

    def load_pre_weight(self, path):
        print(" [*] Reading pre-weight...")
        if not os.path.exists(path):
            print(" [!] No pre-weight found...")
            return
        state = torch.load(path)
        return state
