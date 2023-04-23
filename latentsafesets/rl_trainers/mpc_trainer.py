from latentsafesets.rl_trainers import VAETrainer, SafeSetTrainer, Trainer, ValueTrainer, ConstraintTrainer, GoalIndicatorTrainer, PETSDynamicsTrainer, PETSDynamicsTrainer2#, CBFdotTrainer

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

class CBFdotlatentplanaTrainer(Trainer):
    def __init__(self, env, params, cbfd, loss_plotter):
        self.params = params
        self.cbfd =cbfd
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = params['env']

    def initial_train(self, replay_buffer, update_dir):
        if self.cbfd.trained:
            self.plot(os.path.join(update_dir, "cbfd_start.pdf"), replay_buffer)
            self.plotlatent(os.path.join(update_dir, "cbfdlatent_start.pdf"), replay_buffer)
            return

        log.info('Beginning cbfdot initial optimization')

        self.plotlatentgroundtruth(os.path.join(update_dir, "cbfdgroundtruth.pdf"), replay_buffer)#if not spb, then don't plot
        for i in range(self.params['cbfd_init_iters']):#10000
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
            obs=out_dict['obs']
            #obs = out_dict['obs_relative']
            #print('obs',obs)
            #print('obs.shape',obs.shape)
            #rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']#0 or 1
            if self.params['env']=='push' and self.params['push_cbf_strategy']==2:
                rdo,rdn, hvo,hvn, hvd = out_dict['rdoef'], out_dict['rdnef'],out_dict['hvoef'],out_dict['hvnef'], out_dict['hvdef']  # 0 or 1
            else:
                rdo,rdn, hvo,hvn, hvd = out_dict['rdo'], out_dict['rdn'],out_dict['hvo'],out_dict['hvn'], out_dict['hvd']  # 0 or 1
            #rda=np.concatenate((rdo,action),axis=1)
            #rdal = np.concatenate((obs, action), axis=1)#l for latent
            #print('rdal',rdal)
            #print('rdal.shape',rdal.shape)# (256, 34)
            #loss, info = self.cbfd.update(rdal, hvd, already_embedded=True)#loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            #loss, info = self.cbfd.update(obs,action, hvd, already_embedded=True)  #
            loss, info = self.cbfd.update(obs, hvn, already_embedded=True)  #
            self.loss_plotter.add_data(info)#self.constr.update, not self.update!

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating cbfdot function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "cbfd%d.pdf" % i), replay_buffer)
                self.plotlatent(os.path.join(update_dir, "cbfdlatent%d.pdf" % i), replay_buffer)#nothing is plotted if not spb
                self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased%d-11.pdf" % i), replay_buffer,
                                        coeff=1)  # a few lines later
                self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased%d-13.pdf" % i), replay_buffer,coeff=1/3)  # a few lines later
                self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased%d-14.pdf" % i), replay_buffer,
                                        coeff=1 / 4)  # a few lines later
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.cbfd.save(os.path.join(update_dir, 'cbfd_%d.pth' % i))

        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def update(self, replay_buffer, update_dir):
        if self.params['train_cbf']=='no':
            log.info('No episodic cbf dot update optimization!')
        else:
            if self.params['train_cbf']=='no2':
                self.params['cbfd_lr']=0
                log.info('No episodic cbf dot update optimization but show loss on new data!')
            else:
                log.info('Beginning cbf dot update optimization!')
            
            #log.info('cbfd_lr: %f'%(self.params['cbfd_lr']))
            for _ in trange(self.params['cbfd_update_iters']):
                out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
                #next_obs, constr = out_dict['next_obs'], out_dict['constraint']
                #obs, rdo, action, hvd = out_dict['obs'], out_dict['rdo'], out_dict['action'], out_dict['hvd']  # 0 or 1
                if self.params['env']=='push' and self.params['push_cbf_strategy']==2:
                    obs, rdn, hvn = out_dict['obs'], out_dict['rdnef'], out_dict['hvnef']  # 0 or 1
                else:
                    obs, rdn, hvn = out_dict['obs'], out_dict['rdn'], out_dict['hvn']  # 0 or 1
                #rdo,rdn, hvo,hvn, hvd = out_dict['rdoef'], out_dict['rdnef'],out_dict['hvoef'],out_dict['hvnef'], out_dict['hvdef']  # 0 or 1
                #rdo,rdn, hvo,hvn, hvd = out_dict['rdo'], out_dict['rdn'],out_dict['hvo'],out_dict['hvn'], out_dict['hvd']  # 0 or 1
                #obs, rdn, hvn = out_dict['obs_relative'], out_dict['rdn'], out_dict['hvn']  # 0 or 1
                #print('rdo.shape',rdo.shape)#(256, 2)
                #print('action.shape',action.shape)#(256, 2)
                #rda = np.concatenate((rdo, action),axis=1)
                #rdal = np.concatenate((obs, action), axis=1)
                #print('rda.shape',rda.shape)#(256, 4)
                #loss, info = self.constr.update(next_obs, constr, already_embedded=True)
                #loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
                #loss, info = self.cbfd.update(rdal, hvd, already_embedded=True)#if already_embedded is set to false, then the current setting will run into bug
                loss, info = self.cbfd.update(obs, hvn, already_embedded=True)  #
                self.loss_plotter.add_data(info)

            log.info('Creating cbf dot function heatmap')
            self.loss_plotter.plot()
            self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer)#this is using plan a
            #self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
            #self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased.pdf" ), replay_buffer, coeff=1)
            #self.plotconly(os.path.join(update_dir, "cbfdcircle.pdf"), replay_buffer)  # a few lines later
            self.plotlatent(os.path.join(update_dir, "cbfdlatent.pdf"), replay_buffer)
            self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased-11.pdf"), replay_buffer,
                                    coeff=1)  # a few lines later
            self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased-13.pdf"), replay_buffer,coeff=1/3)  # a few lines later
            self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased-14.pdf"), replay_buffer,
                                    coeff=1 / 4)  # a few lines later
            #self.plotlatentgroundtruth(os.path.join(update_dir, "cbfdgroundtruth.pdf"), replay_buffer)

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        #next_obs = out_dict['next_obs_relative']  # rdo = out_dict['rdo']
        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plotconly(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']
        rdo = out_dict['rdo']
        pu.visualize_cbfdotconly(next_obs, self.cbfd,
                             file,
                             env=self.env)


    def plotlatentunbiased(self, file, replay_buffer,coeff):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']
        #next_obs = out_dict['next_obs_relative']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdotlatentunbiased(next_obs, self.cbfd,
                             file,
                             env=self.env,coeff=coeff)

    def plotlatentgroundtruth(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']
        #next_obs = out_dict['next_obs_relative']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdotlatentgroundtruth(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plotlatent(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']
        #next_obs = out_dict['next_obs_relative']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdotlatent(next_obs, self.cbfd,
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

        self.trainers = []#the following shows the sequence of training

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

    def initial_train(self, replay_buffer):#by default the replay buffer is the encoded version
        update_dir = os.path.join(self.logdir, 'initial_train')#create that folder!
        os.makedirs(update_dir, exist_ok=True)#mkdir is here!
        for trainer in self.trainers:#type() method returns class type of the argument(object) passed as parameter
            if type(trainer) == VAETrainer:#VAE is trained totally on images from that folder, no use of replay_buffer
                trainer.initial_train(self.encoder_data_loader, update_dir)
            else:#then it means that the VAE has been trained!
                trainer.initial_train(replay_buffer, update_dir)

    def update(self, replay_buffer, update_num):#the update folder!
        update_dir = os.path.join(self.logdir, 'update_%d' % update_num)
        os.makedirs(update_dir, exist_ok=True)
        for trainer in self.trainers:
            trainer.update(replay_buffer, update_dir)
