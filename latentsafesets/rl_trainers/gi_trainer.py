from .trainer import Trainer
import latentsafesets.utils.plot_utils as pu

import logging
from tqdm import trange
import os
import numpy as np
log = logging.getLogger("gi train")


class GoalIndicatorTrainer(Trainer):
    def __init__(self, env, params, gi, loss_plotter):
        self.params = params
        self.gi = gi#class GoalIndicator
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = params['env']

    def initial_train(self, replay_buffer, update_dir):
        if self.gi.trained:
            self.plot(os.path.join(update_dir, "gi_start.pdf"), replay_buffer)
            return

        log.info('Beginning goal indicator initial optimization')

        for i in range(self.params['gi_init_iters']):#10000
            out_dict = replay_buffer.sample(self.params['gi_batch_size'])#256#get 1 step
            next_obs, rew = out_dict['next_obs'], out_dict['reward']#0/goal or -1/not goal
            #next_obs, rew = out_dict['next_obs_relative'], out_dict['reward']  # 0/goal or -1/not goal

            loss, info = self.gi.update(next_obs, rew, already_embedded=True)
            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating goal indicator function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "gi%d.pdf" % i), replay_buffer)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.gi.save(os.path.join(update_dir, 'gi_%d.pth' % i))

        # spbu.evaluate_constraint_func(self.gi, file=os.path.join(update_dir, "gi_init.pdf"))
        self.gi.save(os.path.join(update_dir, 'gi.pth'))

    def update(self, replay_buffer, update_dir):
        log.info('Beginning goal indicator update optimization')

        for _ in trange(self.params['gi_update_iters']):
            out_dict = replay_buffer.sample(self.params['gi_batch_size'])
            next_obs, rew = out_dict['next_obs'], out_dict['reward']
            #next_obs, rew = out_dict['next_obs_relative'], out_dict['reward']  # 0/goal or -1/not goal

            loss, info = self.gi.update(next_obs, rew, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating goal indicator function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "gi.pdf"), replay_buffer)
        self.gi.save(os.path.join(update_dir, 'gi.pth'))

    def initial_train_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
        if self.gi.trained:
            self.plot(os.path.join(update_dir, "gi_start.pdf"), replay_buffer_success)
            return

        log.info('Beginning goal indicator initial optimization')

        for i in range(self.params['gi_init_iters']):#10000
            ratio=0.7#0.75#
            successbatch=int(ratio*self.params['dyn_batch_size'])
            out_dict = replay_buffer_success.sample(successbatch)#(self.params['gi_batch_size'])#256#get 1 step
            next_obs, rew = out_dict['next_obs'], out_dict['reward']#0/goal or -1/not goal
            #next_obs, rew = out_dict['next_obs_relative'], out_dict['reward']  # 0/goal or -1/not goal
            out_dictus = replay_buffer_unsafe.sample(self.params['dyn_batch_size']-successbatch)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
            #obsus=out_dictus['obs']#us means unsafe
            next_obsus, rewus = out_dictus['next_obs'], out_dictus['reward']
            next_obs=np.vstack((next_obs,next_obsus))
            #print('rew.shape',rew.shape)#179
            #print('rewus.shape',rewus.shape)#77
            rew=np.concatenate((rew,rewus))
            shuffleind=np.random.permutation(next_obs.shape[0])
            next_obs=next_obs[shuffleind]
            rew=rew[shuffleind]
            loss, info = self.gi.update(next_obs, rew, already_embedded=True)
            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating goal indicator function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "gi%d.pdf" % i), replay_buffer_success)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.gi.save(os.path.join(update_dir, 'gi_%d.pth' % i))

        # spbu.evaluate_constraint_func(self.gi, file=os.path.join(update_dir, "gi_init.pdf"))
        self.gi.save(os.path.join(update_dir, 'gi.pth'))

    def update_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
        log.info('Beginning goal indicator update optimization')

        for _ in trange(self.params['gi_update_iters']):
            ratio=0.7#0.75#
            successbatch=int(ratio*self.params['dyn_batch_size'])
            out_dict = replay_buffer_success.sample(successbatch)#(self.params['gi_batch_size'])#
            next_obs, rew = out_dict['next_obs'], out_dict['reward']
            #next_obs, rew = out_dict['next_obs_relative'], out_dict['reward']  # 0/goal or -1/not goal
            out_dictus = replay_buffer_unsafe.sample(self.params['dyn_batch_size']-successbatch)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
            #obsus=out_dictus['obs']#us means unsafe
            next_obsus, rewus = out_dictus['next_obs'], out_dictus['reward']
            #print('next_obs.shape',next_obs.shape)#179,32
            #print('next_obsus.shape',next_obsus.shape)#77,32
            next_obs=np.vstack((next_obs,next_obsus))
            #print('rew.shape',rew.shape)#179
            #print('rewus.shape',rewus.shape)#77
            rew=np.concatenate((rew,rewus))
            shuffleind=np.random.permutation(next_obs.shape[0])
            next_obs=next_obs[shuffleind]
            rew=rew[shuffleind]
            loss, info = self.gi.update(next_obs, rew, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating goal indicator function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "gi.pdf"), replay_buffer_success)
        self.gi.save(os.path.join(update_dir, 'gi.pth'))

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['constr_batch_size'])
        next_obs = out_dict['next_obs']
        #next_obs = out_dict['next_obs_relative']
        pu.visualize_onezero(next_obs, self.gi,
                             file,
                             env=self.env)
