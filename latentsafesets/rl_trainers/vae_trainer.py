from .trainer import Trainer
import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.modules import VanillaVAE
import latentsafesets.utils as utils

import torch
from torchvision.utils import save_image
import numpy as np

import logging
import os

import torchvision.transforms as transforms

log = logging.getLogger("dyn train")


class VAETrainer(Trainer):
    def __init__(self, params, vae: VanillaVAE, loss_plotter):
        self.params = params#get the parameters for the VAE
        self.vae = vae#get the class of the VAE
        self.loss_plotter = loss_plotter

        self.frame_stack = params['frame_stack']#false in spb, true in push
        self.d_latent = params['d_latent']#32

    def initial_train(self, enc_data_loader, update_dir, force_train=False):
        if self.vae.trained and not force_train:
            return

        log.info('Beginning vae initial optimization')

        for i in range(self.params['enc_init_iters']):#100k default
            obs = enc_data_loader.sample(self.params['enc_batch_size'])#256#256 images!

            loss, info = self.vae.update(obs)
            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:#default 100
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:#500
                log.info('Creating vae visualizaton')
                self.loss_plotter.plot()
                self.plot_vae(obs, update_dir, i=i)
                states=np.random.randint(32, size=2)
                #self.plot_vaetransform(obs,states, update_dir, i=i)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:#2000
                self.vae.save(os.path.join(update_dir, 'vae_%d.pth' % i))

        self.vae.save(os.path.join(update_dir, 'vae.pth'))#that is the real last one!

    def initial_train_cbf(self, enc_data_loader, update_dir, force_train=False):#plan B!
        if self.vae.trained and not force_train:
            return

        #env_name = self.params['env']  # spb
        #frame_stack = self.params['frame_stack']  # 1 or not? In spb is is not stacking
        demo_trajectories = []
        for count, data_dir in list(zip(self.params['data_counts'], self.params['data_dirs'])):  # see 180-200
            # demo_trajectories += utils.load_trajectories(count, file='data/' + data_dir)#98#SimplePointBot
            demo_trajectories += utils.load_trajectories(count, file='data_relative/' + data_dir)  # 98
            #demo_trajectories += utils.load_trajectories_relative(count, file='data_relative/' + data_dir)  # 98
            # count=50, each for non-constraint and constraint, thus together 100
        #print('len(demo_trajectories)',len(demo_trajectories))#100
        #print('size(demo_trajectories)',size(demo_trajectories))#size not defined?

        #i = 0
        log.info('Beginning vae initial optimization')

        for i in range(self.params['enc_init_iters']):#100k default
            #obs,dist = enc_data_loader.sample(self.params['enc_batch_size'])#256#256 images!
            obs, dist = enc_data_loader.sample_cbf(self.params['enc_batch_size'],demo_trajectories)  # 256#256 images!#plan B!
            loss, info = self.vae.update_cbf(obs,dist)
            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:#default 100
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:#500
                log.info('Creating vae visualizaton')
                self.loss_plotter.plot()
                self.plot_vae(obs, update_dir, i=i)
                #states=np.random.randint(32, size=2)
                #self.plot_vaetransform(obs,states, update_dir, i=i)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:#2000
                self.vae.save(os.path.join(update_dir, 'vae_%d.pth' % i))

        self.vae.save(os.path.join(update_dir, 'vae.pth'))#that is the real last one!

    def update(self, replay_buffer, update_dir):
        pass#it means that for the VAE the initial train is enough to achieve the aim

    def update_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
        pass#it means that for the VAE the initial train is enough to achieve the aim

    def update_m2_withonline(self, replay_buffer_success, update_dir,replay_buffer_unsafe,rso,ruo):
        pass#it means that for the VAE the initial train is enough to achieve the aim

    def plot_vae(self, obs, update_dir, i=0):#plot that figure!
        if self.frame_stack == 1:
            obs = np.array([np.array(im).transpose((2, 0, 1)) for im in obs]) / 255
        else:
            obs = ptu.torchify(np.array(
                [[np.array(im).transpose((2, 0, 1)) for im in stack] for stack in obs]
            )) / 255

        with torch.no_grad():#
            sample = torch.randn(64, self.d_latent).to(ptu.TORCH_DEVICE)#not specific to the observation!
            sample = self.vae.decode(sample).cpu()#the above 64 has no special meaning! just to sample 64 images!
            if self.frame_stack > 1:
                # Sample n index randomely
                arange = torch.arange(64)
                ind = arange // 22#what is this operation?
                sample = sample[arange, ind]
            save_image(sample, os.path.join(update_dir, 'sample_%d.png' % i))

        with torch.no_grad():
            data = ptu.torchify(obs[:8])#transform the numpy array to torch tensors
            recon = self.vae.decode(self.vae.encode(data))#8 means you just plot 8 figures/plots
            if self.frame_stack > 1:
                ls = []
                for j in range(self.frame_stack):
                    ls.append(data[:, j])
                    ls.append(recon[:, j].view(8, 3, 64, 64))
                comparison = torch.cat(ls)
            else:
                comparison = torch.cat([data,
                                        recon.view(8, 3, 64, 64)])
            save_image(comparison.cpu(), os.path.join(update_dir, 'recon_%d.png' % i))

    def plot_vaetransform(self, obs, states, update_dir, i=0):#plot that figure!
        if self.frame_stack == 1:
            #obs = np.array([np.array(im).transpose((2, 0, 1)) for im in obs]) / 255
            obs = np.array(
                [np.array(transforms.functional.affine(im, 0, (states[0],states[1]), 1, 0)).transpose((2, 0, 1)) for im in obs])/255
            #obs=ptu.torchify(obs)
            #print('obs',obs)#device='cuda:0'
            #print('obs.shape',obs.shape)#obs.shape torch.Size([256, 3, 64, 64])
            #transform=transforms.functional.affine(obs,0,(0.4, 0.6),1,0)

            #obs = ptu.torchify(obs)
            #obs=transform(obs)
            #obs = [transforms.functional.affine(im,0,(0.4, 0.6),1,0) for im in obs]#plot(affine_imgs)
        else:
            obs = ptu.torchify(np.array(
                [[np.array(im).transpose((2, 0, 1)) for im in stack] for stack in obs]
            )) / 255

        with torch.no_grad():#
            sample = torch.randn(64, self.d_latent).to(ptu.TORCH_DEVICE)#not specific to the observation!
            sample = self.vae.decode(sample).cpu()#the above 64 has no special meaning! just to sample 64 images!
            if self.frame_stack > 1:
                # Sample n index randomely
                arange = torch.arange(64)
                ind = arange // 22#what is this operation?
                sample = sample[arange, ind]
            save_image(sample, os.path.join(update_dir, 'sampletransform_%d.png' % i))

        with torch.no_grad():
            data = ptu.torchify(obs[:8])#transform the numpy array to torch tensors
            recon = self.vae.decode(self.vae.encode(data))#8 means you just plot 8 figures/plots
            if self.frame_stack > 1:
                ls = []
                for j in range(self.frame_stack):
                    ls.append(data[:, j])
                    ls.append(recon[:, j].view(8, 3, 64, 64))
                comparison = torch.cat(ls)
            else:
                comparison = torch.cat([data,
                                        recon.view(8, 3, 64, 64)])
            save_image(comparison.cpu(), os.path.join(update_dir, 'recon_%dtransform.png' % i))