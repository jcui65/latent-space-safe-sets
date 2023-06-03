from .trainer import Trainer
import latentsafesets.utils.plot_utils as pu

import logging
from tqdm import trange
import os
import numpy as np
log = logging.getLogger("constr train")


class ConstraintTrainer(Trainer):
    def __init__(self, env, params, constr, loss_plotter):
        self.params = params
        self.constr = constr
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = params['env']

    def initial_train(self, replay_buffer, update_dir):
        if self.constr.trained:
            self.plot(os.path.join(update_dir, "constr_start.pdf"), replay_buffer)
            return

        log.info('Beginning constraint initial optimization')

        for i in range(self.params['constr_init_iters']):#10000
            out_dict = replay_buffer.sample(self.params['constr_batch_size'])#256
            next_obs, constr = out_dict['next_obs'], out_dict['constraint']#0 or 1
            #next_obs, constr = out_dict['next_obs_relative'], out_dict['constraint']  # 0 or 1

            loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            self.loss_plotter.add_data(info)#self.constr.update, not self.update!

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating constraint function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "constr%d.pdf" % i), replay_buffer)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.constr.save(os.path.join(update_dir, 'constr_%d.pth' % i))

        self.constr.save(os.path.join(update_dir, 'constr.pth'))

    def update(self, replay_buffer, update_dir):
        log.info('Beginning constraint update optimization')

        for _ in trange(self.params['constr_update_iters']):
            out_dict = replay_buffer.sample(self.params['constr_batch_size'])
            next_obs, constr = out_dict['next_obs'], out_dict['constraint']
            #next_obs, constr = out_dict['next_obs_relative'], out_dict['constraint']

            loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating constraint function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "constr.pdf"), replay_buffer)
        self.constr.save(os.path.join(update_dir, 'constr.pth'))

    def initial_train_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
        if self.constr.trained:
            self.plot(os.path.join(update_dir, "constr_start.pdf"), replay_buffer)
            return

        log.info('Beginning constraint initial optimization')

        for i in range(self.params['constr_init_iters']):#10000
            out_dict = replay_buffer_success.sample(self.params['constr_batch_size'])#256
            next_obs, constr = out_dict['next_obs'], out_dict['constraint']#0 or 1
            #next_obs, constr = out_dict['next_obs_relative'], out_dict['constraint']  # 0 or 1

            loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            self.loss_plotter.add_data(info)#self.constr.update, not self.update!

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating constraint function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "constr%d.pdf" % i), replay_buffer)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.constr.save(os.path.join(update_dir, 'constr_%d.pth' % i))

        self.constr.save(os.path.join(update_dir, 'constr.pth'))

    def update_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
        log.info('Beginning constraint update optimization')

        for _ in trange(self.params['constr_update_iters']):
            out_dict = replay_buffer_success.sample(self.params['constr_batch_size'])
            next_obs, constr = out_dict['next_obs'], out_dict['constraint']
            #next_obs, constr = out_dict['next_obs_relative'], out_dict['constraint']

            out_dictus = replay_buffer_unsafe.sample(self.params['dyn_batch_size']-successbatch)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
            #obsus=out_dictus['obs']#us means unsafe
            next_obsus, construs = out_dictus['next_obs'], out_dictus['constraint']
            next_obs=np.vstack((next_obs,next_obsus))
            constr=np.vstack((constr,construs))
            shuffleind=np.random.permutation(next_obs.shape[0])
            next_obs=next_obs[shuffleind]
            rew=rew[shuffleind]

            loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating constraint function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "constr.pdf"), replay_buffer)
        self.constr.save(os.path.join(update_dir, 'constr.pth'))

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['constr_batch_size'])
        next_obs = out_dict['next_obs']
        #next_obs = out_dict['next_obs_relative']
        pu.visualize_onezero(next_obs, self.constr,
                             file,
                             env=self.env)
