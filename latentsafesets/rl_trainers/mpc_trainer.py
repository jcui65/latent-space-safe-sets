from latentsafesets.rl_trainers import VAETrainer, SafeSetTrainer, Trainer, ValueTrainer, ConstraintTrainer, GoalIndicatorTrainer, PETSDynamicsTrainer,CBFdotlatentplanaTrainer#, PETSDynamicsTrainer2#, CBFdotTrainer

from latentsafesets.utils import LossPlotter, EncoderDataLoader

import os

import numpy as np

from .trainer import Trainer
import latentsafesets.utils.plot_utils as pu

import logging
from tqdm import trange
import os

log = logging.getLogger("cbfd train")


class CBFdotTrainer(Trainer):
    def __init__(self, env, params, cbfd, loss_plotter):
        self.params = params
        self.cbfd =cbfd
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = params['env']

    def initial_train(self, replay_buffer, update_dir):
        if self.cbfd.trained:
            self.plot(os.path.join(update_dir, "cbfd_start.pdf"), replay_buffer)
            return

        log.info('Beginning cbfdot initial optimization')

        for i in range(self.params['cbfd_init_iters']):#10000
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']#0 or 1
            rda=np.concatenate((rdo,action),axis=1)
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            self.loss_plotter.add_data(info)#self.constr.update, not self.update!

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating cbfdot function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "cbfd%d.pdf" % i), replay_buffer)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.cbfd.save(os.path.join(update_dir, 'cbfd_%d.pth' % i))

        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def update(self, replay_buffer, update_dir):
        log.info('Beginning cbf dot update optimization!')

        for _ in trange(self.params['cbfd_update_iters']):
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
            #next_obs, constr = out_dict['next_obs'], out_dict['constraint']
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']  # 0 or 1
            #print('rdo.shape',rdo.shape)#(256, 2)
            #print('action.shape',action.shape)#(256, 2)
            rda = np.concatenate((rdo, action),axis=1)
            #print('rda.shape',rda.shape)#(256, 4)
            #loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating cbf dot function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer)
        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
        self.plotconly(os.path.join(update_dir, "cbfdcircle.pdf"), replay_buffer)  # a few lines later

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plotconly(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdotconly(next_obs, self.cbfd,
                             file,
                             env=self.env)


class CBFdotlatentTrainer(Trainer):
    def __init__(self, env, params, cbfd, loss_plotter):
        self.params = params
        self.cbfd =cbfd
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = params['env']

    def initial_train(self, replay_buffer, update_dir):
        if self.cbfd.trained:
            self.plot(os.path.join(update_dir, "cbfdlatent_start.pdf"), replay_buffer)
            return

        log.info('Beginning cbfdot initial optimization')

        for i in range(self.params['cbfd_init_iters']):#10000
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
            obs=out_dict['obs']
            #print('obs',obs)
            #print('obs.shape',obs.shape)
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']#0 or 1
            #rda=np.concatenate((rdo,action),axis=1)
            #rdal = np.concatenate((obs, action), axis=1)#l for latent
            #print('rdal',rdal)
            #print('rdal.shape',rdal.shape)# (256, 34)
            #loss, info = self.cbfd.update(rdal, hvd, already_embedded=True)#loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            loss, info = self.cbfd.update(obs,action, hvd, already_embedded=True)  #
            self.loss_plotter.add_data(info)#self.constr.update, not self.update!

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating cbfdot function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "cbfd%d.pdf" % i), replay_buffer)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.cbfd.save(os.path.join(update_dir, 'cbfd_%d.pth' % i))

        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def update(self, replay_buffer, update_dir):
        log.info('Beginning cbf dot update optimization!')

        for _ in trange(self.params['cbfd_update_iters']):
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
            #next_obs, constr = out_dict['next_obs'], out_dict['constraint']
            obs, rdo, action, hvd = out_dict['obs'], out_dict['rdo'], out_dict['action'], out_dict['hvd']  # 0 or 1
            #print('rdo.shape',rdo.shape)#(256, 2)
            #print('action.shape',action.shape)#(256, 2)
            #rda = np.concatenate((rdo, action),axis=1)
            #rdal = np.concatenate((obs, action), axis=1)
            #print('rda.shape',rda.shape)#(256, 4)
            #loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            #loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            #loss, info = self.cbfd.update(rdal, hvd, already_embedded=True)#if already_embedded is set to false, then the current setting will run into bug
            loss, info = self.cbfd.update(obs,action, hvd, already_embedded=True)  #
            self.loss_plotter.add_data(info)

        log.info('Creating cbf dot function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer)
        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
        #self.plotconly(os.path.join(update_dir, "cbfdcircle.pdf"), replay_buffer)  # a few lines later

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plotconly(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdotconly(next_obs, self.cbfd,
                             file,
                             env=self.env)


class MPCTrainer(Trainer):

    def __init__(self, env, params, modules):

        self.params = params
        self.env = env
        #self.logdir = params['logdir']
        self.logdirbeforeseed = params['logdir']
        seed=self.params['seed']#should I keep self?
        self.logdir=os.path.join(self.logdirbeforeseed, str(seed))
        #loss_plotter = LossPlotter(os.path.join(params['logdir'], 'loss_plots'))
        loss_plotter = LossPlotter(os.path.join(self.logdir, 'loss_plots'))
        self.encoder_data_loader = EncoderDataLoader(params)
        light=params['light']
        self.online=params['online']
        self.trainers = []#the following shows the sequence of training
        self.ways=params['ways']
        self.trainers.append(VAETrainer(params, modules['enc'], loss_plotter))
        self.trainers.append(PETSDynamicsTrainer(params, modules['dyn'], loss_plotter))
        self.trainers.append(ValueTrainer(env, params, modules['val'], loss_plotter))
        self.trainers.append(GoalIndicatorTrainer(env, params, modules['gi'], loss_plotter))
        if light=='normal' or light=='ls3':
            self.trainers.append(SafeSetTrainer(env, params, modules['ss'], loss_plotter))
        if light=='normal' or light=='ls3':
            self.trainers.append(ConstraintTrainer(env, params, modules['constr'], loss_plotter))
            #self.trainers.append(CBFdotTrainer(env, params, modules['cbfd'], loss_plotter))
            #self.trainers.append(CBFdotlatentTrainer(env, params, modules['cbfd'], loss_plotter))  
        if light=='normal' or light=='light':
            self.trainers.append(CBFdotlatentplanaTrainer(env, params, modules['cbfd'], loss_plotter))
        #self.trainers.append(PETSDynamicsTrainer2(params, modules['dyn2'], loss_plotter))
        #self.trainers.append(VAETrainer(params, modules['enc2'], loss_plotter))

    def initial_train(self, replay_buffer,replay_buffer_unsafe=None):#by default the replay buffer is the encoded version
        update_dir = os.path.join(self.logdir, 'initial_train')#create that folder!
        os.makedirs(update_dir, exist_ok=True)#mkdir is here!
        for trainer in self.trainers:#type() method returns class type of the argument(object) passed as parameter
            if type(trainer) == VAETrainer:#VAE is trained totally on images from that folder, no use of replay_buffer
                trainer.initial_train(self.encoder_data_loader, update_dir)
            else:#then it means that the VAE has been trained!
                if type(trainer)!=CBFdotlatentplanaTrainer:
                    trainer.initial_train(replay_buffer, update_dir)
                else:
                    trainer.initial_train(replay_buffer, update_dir,replay_buffer_unsafe)

    def initial_train_m2(self, replay_buffer_success,replay_buffer_unsafe):#by default the replay buffer is the encoded version
        update_dir = os.path.join(self.logdir, 'initial_train')#create that folder!
        os.makedirs(update_dir, exist_ok=True)#mkdir is here!
        for trainer in self.trainers:#type() method returns class type of the argument(object) passed as parameter
            if type(trainer) == VAETrainer:#VAE is trained totally on images from that folder, no use of replay_buffer
                trainer.initial_train(self.encoder_data_loader, update_dir)
            elif type(trainer)!=CBFdotlatentplanaTrainer:#then it means that the VAE has been trained!#now both success and unsafe are required!
                trainer.initial_train_m2(replay_buffer_success, update_dir,replay_buffer_unsafe)
            else:
                if self.ways==1:
                    trainer.initial_train_m2(replay_buffer_success, update_dir,replay_buffer_unsafe)
                elif self.ways==2:
                    trainer.initial_train_m2_0109(replay_buffer_success, update_dir,replay_buffer_unsafe)

    def update(self, replay_buffer, update_num,replay_buffer_unsafe=None):#the update folder!
        update_dir = os.path.join(self.logdir, 'update_%d' % update_num)
        os.makedirs(update_dir, exist_ok=True)
        episodiccbfdhz=0
        for trainer in self.trainers:
            #trainer.update(replay_buffer, update_dir)
            if type(trainer)!=CBFdotlatentplanaTrainer:
                trainer.update(replay_buffer, update_dir)#pay attention to the details!
            else:
                episodiccbfdhz=trainer.update(replay_buffer, update_dir,replay_buffer_unsafe)
        return episodiccbfdhz

    def update_m2(self, replay_buffer_success, update_num,replay_buffer_unsafe):#,replay_buffer_success_online, replay_buffer_unsafe_online):
        #the update folder!
        update_dir = os.path.join(self.logdir, 'update_%d' % update_num)
        os.makedirs(update_dir, exist_ok=True)
        for trainer in self.trainers:
            #trainer.update(replay_buffer, update_dir)
            if type(trainer)!=CBFdotlatentplanaTrainer:
                trainer.update_m2(replay_buffer_success, update_dir,replay_buffer_unsafe)#pay attention to the details!
            else:
                if self.ways==1:
                    episodiccbfdhz=trainer.update_m2(replay_buffer_success, update_dir,replay_buffer_unsafe)
                elif self.ways==2:
                    episodiccbfdhz=trainer.update_m2_0109(replay_buffer_success, update_dir,replay_buffer_unsafe)
        return episodiccbfdhz#returning dhz only
    
    def update_m2_withonline(self, replay_buffer_success, update_num,replay_buffer_unsafe,replay_buffer_success_online, replay_buffer_unsafe_online):
        #the update folder!
        update_dir = os.path.join(self.logdir, 'update_%d' % update_num)
        os.makedirs(update_dir, exist_ok=True)
        for trainer in self.trainers:
            #trainer.update(replay_buffer, update_dir)
            if type(trainer)!=CBFdotlatentplanaTrainer:
                trainer.update_m2_withonline(replay_buffer_success, update_dir,replay_buffer_unsafe,replay_buffer_success_online, replay_buffer_unsafe_online)#pay attention to the details!
            else:
                if self.ways==1:
                    episodiccbfdhz=trainer.update_m2_withonline(replay_buffer_success, update_dir,replay_buffer_unsafe)
                elif self.ways==2:
                    episodiccbfdhz=trainer.update_m2_0109_withonline(replay_buffer_success, update_dir,replay_buffer_unsafe,replay_buffer_success_online, replay_buffer_unsafe_online)
        return episodiccbfdhz#returning dhz only