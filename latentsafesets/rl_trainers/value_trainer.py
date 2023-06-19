from .trainer import Trainer
import latentsafesets.utils.plot_utils as pu

import logging
from tqdm import trange
import os
import numpy as np
log = logging.getLogger("val train")


class ValueTrainer(Trainer):
    def __init__(self, env, params, value, loss_plotter):
        self.params = params
        self.value = value
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = params['env']
        self.batch_size = params['val_batch_size']#256
        self.ensemble = params['val_ensemble']#false or true
        self.n_models = params['val_n_models'] if params['val_ensemble'] else 0#5 or 0

    def initial_train(self, replay_buffer, update_dir):
        if self.value.trained:
            self.plot(os.path.join(update_dir, "val_start.pdf"), replay_buffer)
            return#I don't see the above thing

        log.info('Beginning value initial optimization')

        for i in range(2 * self.params['val_init_iters']):#2*10000=20000
            if i < self.params['val_init_iters']:#the first 10000 iterations
                out_dict = replay_buffer.sample_positive(self.batch_size, 'on_policy', self.n_models)
                obs, rtg = out_dict['obs'], out_dict['rtg']
                #obs, rtg = out_dict['obs_relative'], out_dict['rtg']

                loss, info = self.value.update_init(obs, rtg, already_embedded=True)
            else:
                out_dict = replay_buffer.sample_positive(self.batch_size, 'on_policy', self.n_models)
                obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], \
                                           out_dict['reward'], out_dict['done']
                #obs, next_obs, rew, done = out_dict['obs_relative'], out_dict['next_obs_relative'], out_dict['reward'], out_dict['done']

                loss, info = self.value.update(obs, rew, next_obs, done, already_embedded=True)
            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:#100
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:#500
                log.info('Creating value function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "val%d.pdf" % i), replay_buffer)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:#2000
                self.value.save(os.path.join(update_dir, 'val_%d.pth' % i))

        self.value.save(os.path.join(update_dir, 'val.pth'))

    def update(self, replay_buffer, update_dir):
        log.info('Beginning value update optimization')

        for _ in trange(self.params['val_update_iters']):#2000
            out_dict = replay_buffer.sample_positive(self.batch_size, 'on_policy', self.n_models)
            obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], out_dict['reward'], \
                                       out_dict['done']
            #obs, next_obs, rew, done = out_dict['obs_relative'], out_dict['next_obs_relative'], out_dict['reward'], out_dict['done']

            loss, info = self.value.update(obs, rew, next_obs, done, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating value function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "val.pdf"), replay_buffer)
        self.value.save(os.path.join(update_dir, 'val.pth'))

    def initial_train_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
        if self.value.trained:
            self.plot(os.path.join(update_dir, "val_start.pdf"), replay_buffer_success)
            return#I don't see the above thing

        log.info('Beginning value initial optimization')

        for i in range(2 * self.params['val_init_iters']):#2*10000=20000
            if i < self.params['val_init_iters']:#the first 10000 iterations
                out_dict = replay_buffer_success.sample_positive(self.batch_size, 'on_policy', self.n_models)
                obs, rtg = out_dict['obs'], out_dict['rtg']
                #obs, rtg = out_dict['obs_relative'], out_dict['rtg']

                loss, info = self.value.update_init(obs, rtg, already_embedded=True)
            else:
                out_dict = replay_buffer_success.sample_positive(self.batch_size, 'on_policy', self.n_models)
                obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], \
                                           out_dict['reward'], out_dict['done']
                #obs, next_obs, rew, done = out_dict['obs_relative'], out_dict['next_obs_relative'], out_dict['reward'], out_dict['done']

                loss, info = self.value.update(obs, rew, next_obs, done, already_embedded=True)
            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:#100
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:#500
                log.info('Creating value function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "val%d.pdf" % i), replay_buffer_success)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:#2000
                self.value.save(os.path.join(update_dir, 'val_%d.pth' % i))

        self.value.save(os.path.join(update_dir, 'val.pth'))

    def update_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
        log.info('Beginning value update optimization')

        for i in trange(self.params['val_update_iters']):#2000

            out_dict = replay_buffer_success.sample_positive(self.params['constr_batch_size'], 'on_policy', self.n_models)
            obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], out_dict['reward'], \
                                       out_dict['done']#just use all safe sample, OK?
            '''
            ratious=1/8
            unsafebatch=ratious*self.params['dyn_batch_size']

            if len(replay_buffer_unsafe)<=unsafebatch:
                successbatch=self.params['dyn_batch_size']
                out_dict = replay_buffer_success.sample_positive(successbatch, 'on_policy', self.n_models)
                obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], out_dict['reward'], \
                                       out_dict['done']
                if i==0:
                    log.info('not yet having online unsafe trajectories!')#pass
            else:
                out_dictus = replay_buffer_unsafe.sample_positive(unsafebatch, 'on_policy', self.n_models)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
                #obsus=out_dictus['obs']#us means unsafe
                obsus, next_obsus, rewus, doneus = out_dictus['obs'], out_dictus['next_obs'], out_dictus['reward'], \
                                       out_dictus['done']
                #ratio=7/8#0.75#0.7#
                successbatch=self.params['dyn_batch_size']-unsafebatch
                out_dict = replay_buffer_success.sample_positive(successbatch, 'on_policy', self.n_models)
                obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], out_dict['reward'], \
                                        out_dict['done']
                obs=np.vstack((obs,obsus))
                next_obs=np.vstack((next_obs,next_obsus))
                rew=np.vstack((rew,rewus))
                done=np.concatenate((done,doneus))
                shuffleind=np.random.permutation(obs.shape[0])
                obs=obs[shuffleind]
                next_obs=next_obs[shuffleind]
                rew=rew[shuffleind]
                done=done[shuffleind]
            '''
            loss, info = self.value.update(obs, rew, next_obs, done, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating value function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "val.pdf"), replay_buffer_success)
        self.value.save(os.path.join(update_dir, 'val.pth'))

    def update_m2_withonline(self, replay_buffer_success, update_dir,replay_buffer_unsafe,replay_buffer_success_online, replay_buffer_unsafe_online):
        log.info('Beginning value update optimization including online buffer!')
        lens=len(replay_buffer_success)
        lenu=len(replay_buffer_unsafe)
        lenso=len(replay_buffer_success_online)
        lenuo=len(replay_buffer_unsafe_online)
        lentotal=lens+lenu+lenso+lenuo
        ratios=lens/lentotal
        ratiou=lenu/lentotal#this is useless in value training
        ratioso=lenso/lentotal
        ratiouo=lenuo/lentotal
        #k=0
        for i in trange(self.params['val_update_iters']):#2000
            successbatch=int(ratios*self.params['dyn_batch_size'])
            #out_dict = replay_buffer_success.sample_positive(self.params['constr_batch_size'], 'on_policy', self.n_models)
            out_dict = replay_buffer_success.sample_positive(successbatch, 'on_policy', self.n_models)
            obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], out_dict['reward'], \
                                       out_dict['done']#just use all safe sample, OK?
            #print('rew.shape',rew.shape)#(5,80)#(5, 80, 32)
            '''
            ratious=1/8
            unsafebatch=ratious*self.params['dyn_batch_size']

            if len(replay_buffer_unsafe)<=unsafebatch:
                successbatch=self.params['dyn_batch_size']
                out_dict = replay_buffer_success.sample_positive(successbatch, 'on_policy', self.n_models)
                obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], out_dict['reward'], \
                                       out_dict['done']
                if i==0:
                    log.info('not yet having online unsafe trajectories!')#pass
            else:
                out_dictus = replay_buffer_unsafe.sample_positive(unsafebatch, 'on_policy', self.n_models)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
                #obsus=out_dictus['obs']#us means unsafe
                obsus, next_obsus, rewus, doneus = out_dictus['obs'], out_dictus['next_obs'], out_dictus['reward'], \
                                       out_dictus['done']
                #ratio=7/8#0.75#0.7#
                successbatch=self.params['dyn_batch_size']-unsafebatch
                out_dict = replay_buffer_success.sample_positive(successbatch, 'on_policy', self.n_models)
                obs, next_obs, rew, done = out_dict['obs'], out_dict['next_obs'], out_dict['reward'], \
                                        out_dict['done']
                obs=np.vstack((obs,obsus))
                next_obs=np.vstack((next_obs,next_obsus))
                rew=np.vstack((rew,rewus))
                done=np.concatenate((done,doneus))
                shuffleind=np.random.permutation(obs.shape[0])
                obs=obs[shuffleind]
                next_obs=next_obs[shuffleind]
                rew=rew[shuffleind]
                done=done[shuffleind]
            '''

            if ratiouo==0:
                successobatch=self.params['dyn_batch_size']-successbatch#int(ratioso*self.params['dyn_batch_size'])
            elif ratiouo>0:#
                unsafeobatch=max(2,int(ratiouo*self.params['dyn_batch_size']))#at least one sample! even more should be added!
                out_dictuso = replay_buffer_unsafe_online.sample_positive(successbatch, 'on_policy', self.n_models)#.sample(unsafeobatch)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
                obsuso,next_obsuso, rewuso,doneuso =out_dictuso['obs'], out_dictuso['next_obs'], out_dictuso['reward'],out_dictuso['done']
                obs=np.concatenate((obs,obsuso),axis=1)
                next_obs=np.concatenate((next_obs,next_obsuso),axis=1)
                rew=np.concatenate((rew,rewuso),axis=1)
                done=np.concatenate((done,doneuso),axis=1)#
                successobatch=self.params['dyn_batch_size']-successbatch-unsafeobatch#

            #successobatch=int(ratioso*self.params['dyn_batch_size'])
            #out_dict = replay_buffer_success_online.sample_positive(self.params['constr_batch_size'], 'on_policy', self.n_models)
            out_dicto = replay_buffer_success_online.sample_positive(successobatch, 'on_policy', self.n_models)
            obso, next_obso, rewo, doneo = out_dicto['obs'], out_dicto['next_obs'], out_dicto['reward'], out_dicto['done']#just use all safe sample, OK?
            #print('rewo.shape',rewo.shape)#(5,176)#(5, 176, 32)
            obs=np.concatenate((obs,obso),axis=1)
            next_obs=np.concatenate((next_obs,next_obso),axis=1)
            rew=np.concatenate((rew,rewo),axis=1)
            done=np.concatenate((done,doneo),axis=1)#I think concatenate should be used, as it is a one dimensional thing!
            #if k==0:
                #log.info('online success buffer has been used as expected!')
            shuffleind=np.random.permutation(next_obs.shape[0])
            obs=obs[shuffleind]
            next_obs=next_obs[shuffleind]
            rew=rew[shuffleind]
            done=done[shuffleind]
            loss, info = self.value.update(obs, rew, next_obs, done, already_embedded=True)
            self.loss_plotter.add_data(info)
            #k+=1

        log.info('Creating value function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "val.pdf"), replay_buffer_success)
        self.value.save(os.path.join(update_dir, 'val.pth'))


    def plot(self, file, replay_buffer):
        obs = replay_buffer.sample(30)['obs']
        #obs = replay_buffer.sample(30)['obs_relative']
        pu.visualize_value(obs, self.value, file=file,
                           env=self.env)
