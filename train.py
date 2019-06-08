import os
import time
import torch
import datetime
import torch.nn as nn
import numpy as np
from sagan.model import Generator, Discriminator
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
from utils.utils import tensor2var, var2numpy, denorm


class Trainer(object):

    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.image_size = config.imsize
        self.model_name = config.model
        self.batchsize = config.batchsize
        self.d_iters = config.d_iters
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        self.max_iter = config.max_iter

        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim

        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.hist_step = config.hist_step
        self.version = config.version

        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(
            config.model_save_path, self.version)

        self.g_attn_channel = (self.g_conv_dim*2, self.g_conv_dim)
        self.d_attn_channel = (self.d_conv_dim*4, self.d_conv_dim*8)

        self.build_model()

        self.use_tensorboard = config.use_tensorboard

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):

        data_iter = iter(self.data_loader)

        fixed_z = tensor2var(self.gen.make_hidden(64))

        start_time = time.time()

        for iteration in range(self.max_iter):
            self.gen.train()
            self.dis.train()

            try:
                real_data, _ = next(data_iter)

            except StopIteration:
                data_iter = iter(self.data_loader)
                real_data, _ = next(data_iter)

            real_images = tensor2var(real_data)
            d_out_real, dr1, dr2 = self.dis(real_images)
            d_loss_real = self.relu(1.0 - d_out_real).mean()

            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images, gf1, gf2 = self.gen(z)
            d_out_fake, df1, df2 = self.dis(fake_images)

            d_loss_fake = self.relu(1.0 + d_out_fake).mean()
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images, gf1, gf2 = self.gen(z)
            g_out_fake, df1, df2 = self.dis(fake_images)

            g_loss_fake = - g_out_fake.mean()
            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            if (iteration + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                d_loss_real = d_loss_real.item()
                d_loss_fake = d_loss_fake.item()
                gen_attn1_gamma = self.gen.attn1.gamma.mean().item()
                gen_attn2_gamma = self.gen.attn2.gamma.mean().item()
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_loss_real: {:.4f}, d_loss_fake:{:.4f}".
                      format(elapsed, iteration + 1,
                             self.max_iter, (iteration + 1),
                             self.max_iter, d_loss_real,
                             d_loss_fake,
                             ))
                if self.use_tensorboard:
                    self.writer.add_scalars("data/dis_loss", {
                        "real": d_loss_real,
                        "fake": d_loss_fake
                    }, iteration+1)
                    self.writer.add_scalars("data/dis_output", {"fake": -g_loss_fake.item(),
                                                                "real": d_out_real.mean().item()
                                                                }, iteration+1)
                    self.writer.add_scalars("data/gamma", {"attn1": gen_attn1_gamma,
                                                           "attn2": gen_attn2_gamma
                                                           }, iteration+1)

            # Sample images
            if (iteration + 1) % self.sample_step == 0:
                self.gen.eval()
                fake_images, attn1, attn2 = self.gen(fixed_z)
                fake_images = denorm(fake_images.data)
                n_row = int(np.sqrt(fake_images.size(0)))
                save_image(fake_images,
                           os.path.join(self.sample_path, '{}_fake.png'.format(iteration + 1)))
                fake_images = make_grid(
                    fake_images, nrow=n_row, normalize=False, range=(-1, 1))
                if self.use_tensorboard:
                    self.writer.add_image(
                        "gen_image", fake_images, iteration+1)

                    if (iteration + 1) % self.hist_step == 0:
                        for name, param in self.dis.named_parameters():
                            self.writer.add_histogram("dis/"+name, param.clone().cpu().detach().numpy(), iteration)
                        for name, param in self.gen.named_parameters():
                            self.writer.add_histogram("gen/"+name, param.clone().cpu().detach().numpy(), iteration)

    def build_model(self):

        self.gen = Generator(self.g_attn_channel,
                             image_size=self.image_size, z_dim=self.z_dim, conv_dim=self.g_conv_dim).cuda()
        self.dis = Discriminator(
            self.d_attn_channel, image_size=self.image_size, conv_dim=self.d_conv_dim).cuda()

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.gen.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.dis.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = nn.CrossEntropyLoss()
        self.relu = nn.ReLU()
        # print networks
        print(self.gen)
        print(self.dis)

    def build_tensorboard(self):
        self.writer = SummaryWriter(self.log_path)

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

# %%
# # if __name__ == "__main__":
import pathlib
import os
dirpath = "./vision/gan/gan-torch/samples/sample"
dirpath = pathlib.Path(dirpath)
print(dirpath)
print(pathlib.Path(os.curdir).resolve())
images = dirpath.glob("*.png")
# print(len(list(images)))
for image in sorted(images):
    image_name = os.path.basename(image)

    if int(image_name[:-9]) > 100000 and int(image_name[:-9]) % 5000 != 0:
        os.remove(image)
        continue
    if int(image_name[:-9]) > 300000 and int(image_name[:-9]) % 10000 != 0:
        os.remove(image)
        continue
    image_name_new = "{:07}_fake.png".format(int(image_name[:-9]))
    # image_name = 
    
    os.rename(image, os.path.dirname(image)+"/"+image_name_new)

    # print(list(len(images)))
    # for image in images:
    #     break

