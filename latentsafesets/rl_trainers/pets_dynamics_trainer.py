from .trainer import Trainer
import latentsafesets.utils.spb_utils as spbu
import latentsafesets.utils.plot_utils as pu

import logging
from tqdm import trange
import os
import numpy as np
log = logging.getLogger("dyn train")


class PETSDynamicsTrainer(Trainer):
    def __init__(self, params, dynamics, loss_plotter):
        self.params = params
        self.dynamics = dynamics
        self.loss_plotter = loss_plotter
        self.batchsize=self.params['dyn_batch_size']
        self.ensemble = params['dyn_n_models']#5 by default

        self.env_name = params['env']#spb/reacher/push
        

    def initial_train(self, replay_buffer, update_dir):#update_dir is the folder ended in initial_train
        if self.dynamics.trained:
            self.visualize(os.path.join(update_dir, "dyn_start.gif"), replay_buffer)
            return

        log.info('Beginning dynamics initial optimization')

        for i in range(self.params['dyn_init_iters']):#10000
            out_dict = replay_buffer.sample(self.params['dyn_batch_size'],#256#get sub-dict of corresponding indices
                                            ensemble=self.ensemble)#59 in replay_buffer_encoded
            #print('out_dict',out_dict)
            obs, next_obs, act = out_dict['obs'], out_dict['next_obs'], out_dict['action']#get values of those indices
            #print('obs.shape', obs.shape)#(5, 256, 32)#
            #print('next_obs.shape', next_obs.shape)#(5, 256, 32)#
            #print('act1.shape', act.shape)#(5, 256, 2)#
            #obs, next_obs, act = out_dict['obs_relative'], out_dict['next_obs_relative'], out_dict['action']  # get values of those indices
            #print('obs_relative.shape',obs.shape) #now (5, 256, 32)##(5, 256, 3, 64, 64)#
            #print('next_obs_relative.shape', next_obs.shape)#now (5, 256, 32)#
            #print('act2.shape',act.shape)#(5, 256, 2)#
            #this is the update of self.dynamics, rather than the update of self!
            loss, info = self.dynamics.update(obs, next_obs, act, already_embedded=True)

            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:#100
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:#500
                log.info('Creating dynamics visualization')
                self.loss_plotter.plot()

                self.visualize(os.path.join(update_dir, "dyn%d.gif" % i), replay_buffer)

            if i % self.params['checkpoint_freq'] == 0 and i > 0:#2000
                self.dynamics.save(os.path.join(update_dir, 'dynamics_%d.pth' % i))

        self.dynamics.save(os.path.join(update_dir, 'dyn.pth'))

    def update(self, replay_buffer, update_dir):#this's for update0/1... after init train
        log.info('Beginning dynamics optimization')

        for _ in trange(self.params['dyn_update_iters']):#512
            out_dict = replay_buffer.sample(self.params['dyn_batch_size'],
                                            ensemble=self.ensemble)
            obs, next_obs, act = out_dict['obs'], out_dict['next_obs'], out_dict['action']
            #obs, next_obs, act = out_dict['obs_relative'], out_dict['next_obs_relative'], out_dict['action']

            loss, info = self.dynamics.update(obs, next_obs, act, already_embedded=True)
            self.loss_plotter.add_data(info)#the update is just the dynamics update

        log.info('Creating dynamics heatmap')
        self.loss_plotter.plot()
        self.visualize(os.path.join(update_dir, "dyn.gif"), replay_buffer)
        self.dynamics.save(os.path.join(update_dir, 'dyn.pth'))

    def initial_train_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):#update_dir is the folder ended in initial_train
        if self.dynamics.trained:
            self.visualize(os.path.join(update_dir, "dyn_start.gif"), replay_buffer_success)
            return

        log.info('Beginning dynamics initial optimization')

        for i in range(self.params['dyn_init_iters']):#10000
            #nosuccessdata=
            #nounsafedata=
            #ratio=nosuccessdata/(nosuccessdata+nounsafedata)
            #ratio=0.7#0.75#
            lens=len(replay_buffer_success)
            lenu=len(replay_buffer_unsafe)
            ratio=lens/(lens+lenu)
            successbatch=int(ratio*self.params['dyn_batch_size'])
            out_dict = replay_buffer_success.sample(successbatch,#self.params['dyn_batch_size'],#256#get sub-dict of corresponding indices
                                            ensemble=self.ensemble)#59 in replay_buffer_encoded
            #print('out_dict',out_dict)
            obs, next_obs, act = out_dict['obs'], out_dict['next_obs'], out_dict['action']#get values of those indices
            
            out_dictus = replay_buffer_unsafe.sample(self.params['dyn_batch_size']-successbatch,ensemble=self.ensemble)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
            obsus=out_dictus['obs']#us means unsafe
            obsus, next_obsus, actus = out_dictus['obs'], out_dictus['next_obs'], out_dictus['action']
            obs=np.concatenate((obs,obsus),axis=1)
            next_obs=np.concatenate((next_obs,next_obsus),axis=1)
            #print('hvnold.shape',hvn.shape)
            act=np.concatenate((act,actus),axis=1)#pay attention to the dimension!
            #print('hvnnew.shape',hvn.shape)
            shuffleind=np.random.permutation(obs.shape[0])
            obs=obs[shuffleind]
            next_obs=next_obs[shuffleind]
            act=act[shuffleind]
            
            
            loss, info = self.dynamics.update(obs, next_obs, act, already_embedded=True)

            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:#100
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:#500
                log.info('Creating dynamics visualization')
                self.loss_plotter.plot()

                self.visualize(os.path.join(update_dir, "dyn%d.gif" % i), replay_buffer_success)

            if i % self.params['checkpoint_freq'] == 0 and i > 0:#2000
                self.dynamics.save(os.path.join(update_dir, 'dynamics_%d.pth' % i))

        self.dynamics.save(os.path.join(update_dir, 'dyn.pth'))

    def update_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):#this's for update0/1... after init train
        log.info('Beginning dynamics optimization')

        for _ in trange(self.params['dyn_update_iters']):#512
            ratio=0.7#0.75#
            successbatch=int(ratio*self.params['dyn_batch_size'])
            out_dict = replay_buffer_success.sample(successbatch,#self.params['dyn_batch_size'],
                                            ensemble=self.ensemble)#this is dyn trainer specific?
            obs, next_obs, act = out_dict['obs'], out_dict['next_obs'], out_dict['action']
            #print('act.shape',act.shape)
            #obs, next_obs, act = out_dict['obs_relative'], out_dict['next_obs_relative'], out_dict['action']
            #if replay_buffer_unsafe!=None:
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256

            out_dictus = replay_buffer_unsafe.sample(self.params['dyn_batch_size']-successbatch,ensemble=self.ensemble)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
            #obsus=out_dictus['obs']#us means unsafe

            obsus, next_obsus, actus = out_dictus['obs'], out_dictus['next_obs'], out_dictus['action']
            #print('actus.shape',actus.shape)
            obs=np.concatenate((obs,obsus),axis=1)
            next_obs=np.concatenate((next_obs,next_obsus),axis=1)
            #print('hvnold.shape',hvn.shape)
            act=np.concatenate((act,actus),axis=1)#pay attention to the dimension!
            #print('hvnnew.shape',hvn.shape)
            shuffleind=np.random.permutation(obs.shape[0])
            obs=obs[shuffleind]
            next_obs=next_obs[shuffleind]
            act=act[shuffleind]


            loss, info = self.dynamics.update(obs, next_obs, act, already_embedded=True)
            self.loss_plotter.add_data(info)#the update is just the dynamics update

        log.info('Creating dynamics heatmap')
        self.loss_plotter.plot()
        self.visualize(os.path.join(update_dir, "dyn.gif"), replay_buffer_success)#replay_buffer)#
        self.dynamics.save(os.path.join(update_dir, 'dyn.pth'))#not very high priority!

    def update_m2_withonline(self, replay_buffer_success, update_dir,replay_buffer_unsafe,replay_buffer_success_online, replay_buffer_unsafe_online):#this's for update0/1... after init train
        log.info('Beginning dynamics optimization including online buffer!')
        lens=len(replay_buffer_success)
        lenu=len(replay_buffer_unsafe)
        lenso=len(replay_buffer_success_online)
        lenuo=len(replay_buffer_unsafe_online)
        lentotal=lens+lenu+lenso+lenuo
        ratios=lens/lentotal
        ratiou=lenu/lentotal
        ratioso=lenso/lentotal
        ratiouo=lenuo/lentotal
        k=0
        for _ in trange(self.params['dyn_update_iters']):#512
            #ratio=0.7#0.75#
            successbatch=int(ratios*self.params['dyn_batch_size'])
            out_dict = replay_buffer_success.sample(successbatch,#self.params['dyn_batch_size'],
                                            ensemble=self.ensemble)#this is dyn trainer specific?
            obs, next_obs, act = out_dict['obs'], out_dict['next_obs'], out_dict['action']
            #print('act.shape',act.shape)
            #obs, next_obs, act = out_dict['obs_relative'], out_dict['next_obs_relative'], out_dict['action']
            #if replay_buffer_unsafe!=None:
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
            unsafebatch=int(ratiou*self.params['dyn_batch_size'])
            out_dictus = replay_buffer_unsafe.sample(unsafebatch,ensemble=self.ensemble)#(self.params['dyn_batch_size']-successbatch,ensemble=self.ensemble)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
            #obsus=out_dictus['obs']#us means unsafe

            obsus, next_obsus, actus = out_dictus['obs'], out_dictus['next_obs'], out_dictus['action']
            #print('actus.shape',actus.shape)
            obs=np.concatenate((obs,obsus),axis=1)
            next_obs=np.concatenate((next_obs,next_obsus),axis=1)
            #print('hvnold.shape',hvn.shape)
            act=np.concatenate((act,actus),axis=1)#pay attention to the dimension!
            #print('hvnnew.shape',hvn.shape)
            
            if ratiouo==0:
                successobatch=self.params['dyn_batch_size']-successbatch-unsafebatch#int(ratioso*self.params['dyn_batch_size'])
            elif ratiouo>0:
                unsafeobatch=min(1,int(ratiouo*self.params['dyn_batch_size']))#at least one sample!
                out_dictuso = replay_buffer_unsafe_online.sample(unsafeobatch, ensemble=self.ensemble)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
                obsuso,next_obsuso, actuso = out_dictuso['obs'], out_dictuso['next_obs'], out_dictuso['action']
                obs=np.concatenate((obs,obsuso),axis=1)
                next_obs=np.concatenate((next_obs,next_obsuso),axis=1)
                act=np.concatenate((act,actuso),axis=1)#pay attention to the dimension!
                successobatch=self.params['dyn_batch_size']-successbatch-unsafebatch-unsafeobatch#
            out_dicto = replay_buffer_success_online.sample(successobatch, ensemble=self.ensemble)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
            obso, next_obso, acto = out_dicto['obs'], out_dicto['next_obs'], out_dicto['action']
            obs=np.concatenate((obs,obso),axis=1)
            next_obs=np.concatenate((next_obs,next_obso),axis=1)
            act=np.concatenate((act,acto),axis=1)#pay attention to the dimension!
            if k==0:
                log.info('online success buffer has been used as expected!')
            shuffleind=np.random.permutation(obs.shape[0])
            obs=obs[shuffleind]
            next_obs=next_obs[shuffleind]
            act=act[shuffleind]

            loss, info = self.dynamics.update(obs, next_obs, act, already_embedded=True)
            self.loss_plotter.add_data(info)#the update is just the dynamics update
            k+=1

        log.info('Creating dynamics heatmap')
        self.loss_plotter.plot()
        self.visualize(os.path.join(update_dir, "dyn.gif"), replay_buffer_success)#replay_buffer)#
        self.dynamics.save(os.path.join(update_dir, 'dyn.pth'))#not very high priority!


    def visualize(self, file, replay_buffer):#
        out_dict = replay_buffer.sample_chunk(8, 10)

        obs = out_dict['obs']
        #obs = out_dict['obs_relative']
        act = out_dict['action']
        pu.visualize_dynamics(obs, act, self.dynamics, self.dynamics.encoder, file)

class PETSDynamicsTrainer2(Trainer):
    def __init__(self, params, dynamics, loss_plotter):
        self.params = params
        self.dynamics = dynamics
        self.loss_plotter = loss_plotter

        self.ensemble = params['dyn_n_models']#5 by default

        self.env_name = params['env']#spb/reacher/push

    def initial_train(self, replay_buffer, update_dir):#update_dir is the folder ended in initial_train
        if self.dynamics.trained:
            self.visualize(os.path.join(update_dir, "dyn_start2.gif"), replay_buffer)
            return

        log.info('Beginning dynamics initial optimization')

        for i in range(self.params['dyn_init_iters']):#10000
            out_dict = replay_buffer.sample(self.params['dyn_batch_size'],#256#get sub-dict of corresponding indices
                                            ensemble=self.ensemble)#59 in replay_buffer_encoded
            obs, next_obs, act = out_dict['obs_relative'], out_dict['next_obs_relative'], out_dict['action']  # get values of those indices
            loss, info = self.dynamics.update(obs, next_obs, act, already_embedded=True)

            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:#100
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:#500
                log.info('Creating dynamics visualization')
                self.loss_plotter.plot()

                self.visualize(os.path.join(update_dir, "dyn2%d.gif" % i), replay_buffer)

            if i % self.params['checkpoint_freq'] == 0 and i > 0:#2000
                self.dynamics.save(os.path.join(update_dir, 'dynamics2_%d.pth' % i))

        self.dynamics.save(os.path.join(update_dir, 'dyn2.pth'))

    def update(self, replay_buffer, update_dir):#this's for update0/1... after init train
        log.info('Beginning dynamics optimization')

        for _ in trange(self.params['dyn_update_iters']):#512
            out_dict = replay_buffer.sample(self.params['dyn_batch_size'],
                                            ensemble=self.ensemble)
            #obs, next_obs, act = out_dict['obs'], out_dict['next_obs'], out_dict['action']
            obs, next_obs, act = out_dict['obs_relative'], out_dict['next_obs_relative'], out_dict['action']

            loss, info = self.dynamics.update(obs, next_obs, act, already_embedded=True)
            self.loss_plotter.add_data(info)#the update is just the dynamics update

        log.info('Creating dynamics heatmap')
        self.loss_plotter.plot()
        self.visualize(os.path.join(update_dir, "dyn2.gif"), replay_buffer)
        self.dynamics.save(os.path.join(update_dir, 'dyn2.pth'))

    def visualize(self, file, replay_buffer):
        out_dict = replay_buffer.sample_chunk(8, 10)

        #obs = out_dict['obs']
        obs = out_dict['obs_relative']
        act = out_dict['action']
        pu.visualize_dynamics(obs, act, self.dynamics, self.dynamics.encoder, file)
