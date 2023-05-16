
#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')


import numpy as np

from .trainer import Trainer
import latentsafesets.utils.plot_utils as pu

import logging
from tqdm import trange
import os

log = logging.getLogger("cbfd train")


class CBFdotTrainer(Trainer):#modified from contraint trainer
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

        for i in range(self.params['cbfd_init_iters']):#for example, 10000
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']#0 or 1
            #rdo for relative distance old, hvd for h value difference#print('rdo',rdo)#seems reasonable##print(rdo.shape)#(256,2)#print('action',action)#seems reasonable##print(action.shape)#(256,2)
            rda=np.concatenate((rdo,action),axis=1)#rda for relative distance+action
            #print(rda.shape)#(256,4)#print('hvd',hvd)#seems reasonable##print(hvd.shape)#(256,)
            #rdn,hvo,hvn=out_dict['rdn'], out_dict['hvo'], out_dict['hvn']#
            #print('rdn',rdn)#seems reasonable##print('hvo',hvo)#seems reasonable##print('hvn',hvn)#seems reasonable#
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)#not the update below
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

    def update(self, replay_buffer, update_dir):#this is the update process after initial train
        log.info('Beginning cbf dot update optimization')

        for _ in trange(self.params['cbfd_update_iters']):
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
            #next_obs, constr = out_dict['next_obs'], out_dict['constraint']
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']  # 0 or 1
            rda = np.concatenate((rdo, action),axis=1)

            #loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating cbf dot function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer)#a few lines later
        #self.plotc(os.path.join(update_dir, "cbfdc.pdf"), replay_buffer)  # a few lines later
        self.plotconly(os.path.join(update_dir, "cbfdcircle.pdf"), replay_buffer)  # a few lines later
        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']
        #rdo = out_dict['rdo']
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


class CBFdotplanaTrainer(Trainer):#modified from contraint trainer???
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

        for i in range(self.params['cbfd_init_iters']):#for example, 10000
            if self.params['mean']=='meancbf':
                out_dict = replay_buffer.samplemeancbf(self.params['cbfd_batch_size'])#256
            else:
                out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
            #out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']#0 or 1
            #rdo for relative distance old, hvd for h value difference
            #print('rdo',rdo)#seems reasonable##print(rdo.shape)#(256,2)
            #print('action',action)#seems reasonable##print(action.shape)#(256,2)
            rda=np.concatenate((rdo,action),axis=1)#rda for relative distance+action
            #print(rda.shape)#(256,4)
            #print('hvd',hvd)#seems reasonable##print(hvd.shape)#(256,)
            #rdn,hvo,hvn=out_dict['rdn'], out_dict['hvo'], out_dict['hvn']#
            #print('rdn',rdn)#seems reasonable#
            #print('hvo',hvo)#seems reasonable#
            #print('hvn',hvn)#seems reasonable#
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)#not the update below
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

    def update(self, replay_buffer, update_dir):#this is the update process after initial train
        log.info('Beginning cbf dot update optimization')

        for _ in trange(self.params['cbfd_update_iters']):
            if self.params['mean']=='meancbf':
                out_dict = replay_buffer.samplemeancbf(self.params['cbfd_batch_size'])#256
            else:
                out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
            #out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
            #next_obs, constr = out_dict['next_obs'], out_dict['constraint']
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']  # 0 or 1
            rda = np.concatenate((rdo, action),axis=1)

            #loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating cbf dot function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer)#a few lines later
        #self.plotc(os.path.join(update_dir, "cbfdc.pdf"), replay_buffer)  # a few lines later
        self.plotconly(os.path.join(update_dir, "cbfdcircle.pdf"), replay_buffer)  # a few lines later
        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def plot(self, file, replay_buffer):
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
        #out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plotconly(self, file, replay_buffer):
        #out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdotconly(next_obs, self.cbfd,
                             file,
                             env=self.env)

class CBFdotlatentplanaTrainerold(Trainer):
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
            rdo,rdn, hvo,hvn, hvd = out_dict['rdo'], out_dict['rdn'],out_dict['hvo'],out_dict['hvn'], out_dict['hvd']  # 0 or 1
            loss, info = self.cbfd.update(obs, hvn, already_embedded=True)  #
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
            obs, rdn, hvn = out_dict['obs'], out_dict['rdn'], out_dict['hvn']  # 0 or 1
            loss, info = self.cbfd.update(obs, hvn, already_embedded=True)  #
            self.loss_plotter.add_data(info)

        log.info('Creating cbf dot function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer)
        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
        #self.plotconly(os.path.join(update_dir, "cbfdcircle.pdf"), replay_buffer)  # a few lines later

    def plot(self, file, replay_buffer):
        #out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plotconly(self, file, replay_buffer):
        #out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        pu.visualize_cbfdotconly(next_obs, self.cbfd,
                             file,
                             env=self.env)


class CBFdotlatentplanaTrainer(Trainer):
    def __init__(self, env, params, cbfd, loss_plotter):
        self.params = params
        self.cbfd =cbfd
        self.loss_plotter = loss_plotter
        self.env = env
        self.unsafebuffer=self.params['unsafebuffer']
        if self.unsafebuffer=='yes2':
            self.batchsize=int(self.params['cbfd_batch_size']/2)#
            #log.info('self.batchsize hope 128:%d'% (self.batchsize))#it is 128!
        else:
            self.batchsize=self.params['cbfd_batch_size']#int(self.params['cbfd_batch_size']/2)#
            #log.info('self.batchsize hope 256:%d'% (self.batchsize))
        self.env_name = params['env']

    def initial_train(self, replay_buffer, update_dir,replay_buffer_unsafe):
        if self.cbfd.trained:
            self.plot(os.path.join(update_dir, "cbfd_start.pdf"), replay_buffer,replay_buffer_unsafe)
            self.plotlatent(os.path.join(update_dir, "cbfdlatent_start.pdf"), replay_buffer,replay_buffer_unsafe)
            return

        log.info('Beginning cbfdot initial optimization')

        self.plotlatentgroundtruth(os.path.join(update_dir, "cbfdgroundtruth.pdf"), replay_buffer,replay_buffer_unsafe)#if not spb, then don't plot
        for i in range(self.params['cbfd_init_iters']):#10000
            #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
            if self.params['mean']=='meancbf':
                out_dict = replay_buffer.samplemeancbf(self.batchsize)#sanity check passed!#(self.params['cbfd_batch_size'])#256
                #log.info('training the mean version of the CBF!')
            else:
                out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            obs=out_dict['obs']
            #obs = out_dict['obs_relative']
            #print('obs',obs)
            #print('obs.shape',obs.shape)
            #rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']#0 or 1
            if self.params['env']=='push' and self.params['push_cbf_strategy']==2:
                rdo,rdn, hvo,hvn, hvd = out_dict['rdoef'], out_dict['rdnef'],out_dict['hvoef'],out_dict['hvnef'], out_dict['hvdef']  # 0 or 1
            else:
                rdo,rdn, hvo,hvn, hvd = out_dict['rdo'], out_dict['rdn'],out_dict['hvo'],out_dict['hvn'], out_dict['hvd']  # 0 or 1
            #print('hvn.shape',hvn.shape)
            if replay_buffer_unsafe!=None:
                #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
                if self.params['mean']=='meancbf':
                    out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    #log.info('training the mean version of the CBF!')#sanity check passed!
                else:
                    out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
                obsus=out_dictus['obs']#us means unsafe
                rdous,rdnus, hvous,hvnus, hvdus = out_dictus['rdo'], out_dictus['rdn'],out_dictus['hvo'],out_dictus['hvn'], out_dictus['hvd']  # 0 or 1
                obs=np.vstack((obs,obsus))
                hvn=np.concatenate((hvn,hvnus))
                shuffleind=np.random.permutation(obs.shape[0])
                obs=obs[shuffleind]
                hvn=hvn[shuffleind]
            #loss, info = self.cbfd.update(rdal, hvd, already_embedded=True)#loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            #loss, info = self.cbfd.update(obs,action, hvd, already_embedded=True)  #
            loss, info = self.cbfd.update(obs, hvn, already_embedded=True)  #
            self.loss_plotter.add_data(info)#self.constr.update, not self.update!

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating cbfdot function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "cbfd%d.pdf" % i), replay_buffer,replay_buffer_unsafe)
                self.plotlatent(os.path.join(update_dir, "cbfdlatent%d.pdf" % i), replay_buffer,replay_buffer_unsafe)#nothing is plotted if not spb
                self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased%d-11.pdf" % i), replay_buffer,replay_buffer_unsafe,
                                        coeff=1)  # a few lines later
                self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased%d-13.pdf" % i), replay_buffer,replay_buffer_unsafe,coeff=1/3)  # a few lines later
                self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased%d-14.pdf" % i), replay_buffer,replay_buffer_unsafe,
                                        coeff=1 / 4)  # a few lines later
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.cbfd.save(os.path.join(update_dir, 'cbfd_%d.pth' % i))

        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def update(self, replay_buffer, update_dir,replay_buffer_unsafe):
        if self.params['train_cbf']=='no':
            log.info('No episodic cbf dot update optimization!')
        else:
            if self.params['train_cbf']=='no2':
                self.params['cbfd_lr']=0
                log.info('No episodic cbf dot update optimization but show loss on new data!')
            else:
                log.info('Beginning cbf dot update optimization!')
            dhzepochave=0
            #log.info('cbfd_lr: %f'%(self.params['cbfd_lr']))
            for _ in trange(self.params['cbfd_update_iters']):#512
                #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#random shuffling is done in this step!
                if self.params['mean']=='meancbf':
                    out_dict = replay_buffer.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                else:
                    out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
                #next_obs, constr = out_dict['next_obs'], out_dict['constraint']
                #obs, rdo, action, hvd = out_dict['obs'], out_dict['rdo'], out_dict['action'], out_dict['hvd']  # 0 or 1
                if self.params['env']=='push' and self.params['push_cbf_strategy']==2:
                    obs, rdn, hvn = out_dict['obs'], out_dict['rdnef'], out_dict['hvnef']  # 0 or 1
                else:
                    obs, rdn, hvn = out_dict['obs'], out_dict['rdn'], out_dict['hvn']  # 0 or 1
                    #print('obsold.shape',obs.shape)(128,32)
                #rdo,rdn, hvo,hvn, hvd = out_dict['rdoef'], out_dict['rdnef'],out_dict['hvoef'],out_dict['hvnef'], out_dict['hvdef']  # 0 or 1
                #rdo,rdn, hvo,hvn, hvd = out_dict['rdo'], out_dict['rdn'],out_dict['hvo'],out_dict['hvn'], out_dict['hvd']  # 0 or 1
                #obs, rdn, hvn = out_dict['obs_relative'], out_dict['rdn'], out_dict['hvn']  # 0 or 1
                #print('obsold.shape',obs.shape)
                if replay_buffer_unsafe!=None:
                    #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
                    if self.params['mean']=='meancbf':
                        #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                        if self.params['boundary']=='no':
                            out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                        elif self.params['boundary']=='yes':
                            out_dictus = replay_buffer_unsafe.sample_boundary_meancbf(self.batchsize,'hvn')#(self.params['cbfd_batch_size'])#256
                    else:
                        out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    obsus=out_dictus['obs']#us means unsafe
                    #print('obsus.shape',obsus.shape)(128,32)
                    rdous,rdnus, hvous,hvnus, hvdus = out_dictus['rdo'], out_dictus['rdn'],out_dictus['hvo'],out_dictus['hvn'], out_dictus['hvd']  # 0 or 1
                    obs=np.vstack((obs,obsus))
                    #print('obsnew.shape',obs.shape)(256,32)
                    #print('hvnold.shape',hvn.shape)
                    hvn=np.concatenate((hvn,hvnus))
                    #print('hvnnew.shape',hvn.shape)
                    shuffleind=np.random.permutation(obs.shape[0])
                    obs=obs[shuffleind]
                    hvn=hvn[shuffleind]
                #loss, info = self.constr.update(next_obs, constr, already_embedded=True)
                #loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
                #loss, info = self.cbfd.update(rdal, hvd, already_embedded=True)#if already_embedded is set to false, then the current setting will run into bug
                loss, info = self.cbfd.update(obs, hvn, already_embedded=True)  #info is a dictionary
                self.loss_plotter.add_data(info)
                cbfloss=info['cbf']#this is the real cbf loss
                if self.env_name=='reacher':
                    dhzepochave+=np.sqrt(cbfloss)#faithfully record it!#np.sqrt(min(loss,10))#over 10 is too crazy!
                elif self.env_name=='push':
                    dhzepochave+=np.sqrt(cbfloss)#
                elif self.env_name=='spb':
                    print('just hold it now!')
            dhzepochave=dhzepochave/self.params['cbfd_update_iters']
            dhzepochave=dhzepochave/1000
            log.info('the average dhz of this epochs: %f'%(dhzepochave))
            if self.params['dynamic_dhz']=='yes':
                if self.env_name=='reacher':
                    deal=min(dhzepochave,1*self.params['dhz'])#will it work as expected?deal for dhz epoch ave legit
                else:
                    deal=min(dhzepochave,1*self.params['dhz'])#will it work as expected?deal for dhz epoch ave legit
            else:
                deal=dhzepochave
            log.info('Creating cbf dot function heatmap')
            self.loss_plotter.plot()
            self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer,replay_buffer_unsafe)#this is using plan a
            #self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
            #self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased.pdf" ), replay_buffer, coeff=1)
            #self.plotconly(os.path.join(update_dir, "cbfdcircle.pdf"), replay_buffer)  # a few lines later
            self.plotlatent(os.path.join(update_dir, "cbfdlatent.pdf"), replay_buffer,replay_buffer_unsafe)
            self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased-11.pdf"), replay_buffer,replay_buffer_unsafe,
                                    coeff=1)  # a few lines later
            self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased-13.pdf"), replay_buffer,replay_buffer_unsafe,coeff=1/3)  # a few lines later
            self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased-14.pdf"), replay_buffer,replay_buffer_unsafe,
                                    coeff=1 / 4)  # a few lines later
            #self.plotlatentgroundtruth(os.path.join(update_dir, "cbfdgroundtruth.pdf"), replay_buffer)
            self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
            return deal

    def plot(self, file, replay_buffer,replay_buffer_unsafe):
        #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        if replay_buffer_unsafe!=None:
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
            if self.params['mean']=='meancbf':
                #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                if self.params['boundary']=='no':
                    out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                elif self.params['boundary']=='yes':
                    out_dictus = replay_buffer_unsafe.sample_boundary_meancbf(self.batchsize,'hvn')#(self.params['cbfd_batch_size'])#256
            else:
                out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            next_obsus = out_dictus['next_obs']#rdo = out_dict['rdo']
            next_obs=np.vstack((next_obs,next_obsus))
        #next_obs = out_dict['next_obs_relative']  # rdo = out_dict['rdo']
        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plotconly(self, file, replay_buffer,replay_buffer_unsafe):
        #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']
        rdo = out_dict['rdo']
        if replay_buffer_unsafe!=None:
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
            if self.params['mean']=='meancbf':
                #out_dictus = replay_buffer_unsafe.samplemeancbf(self.params['cbfd_batch_size'])#256
                if self.params['boundary']=='no':
                    out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                elif self.params['boundary']=='yes':
                    out_dictus = replay_buffer_unsafe.sample_boundary_meancbf(self.batchsize,'hvn')#(self.params['cbfd_batch_size'])#256
            else:
                out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            next_obsus = out_dictus['next_obs']#rdo = out_dict['rdo']
            next_obs=np.vstack((next_obs,next_obsus))
        pu.visualize_cbfdotconly(next_obs, self.cbfd,
                             file,
                             env=self.env)


    def plotlatentunbiased(self, file, replay_buffer,replay_buffer_unsafe,coeff):
        #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']
        #next_obs = out_dict['next_obs_relative']
        #rdo = out_dict['rdo']
        if replay_buffer_unsafe!=None:
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
            if self.params['mean']=='meancbf':
                #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                if self.params['boundary']=='no':
                    out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                elif self.params['boundary']=='yes':
                    out_dictus = replay_buffer_unsafe.sample_boundary_meancbf(self.batchsize,'hvn')#(self.params['cbfd_batch_size'])#256
            else:
                out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            next_obsus = out_dictus['next_obs']#rdo = out_dict['rdo']
            next_obs=np.vstack((next_obs,next_obsus))
        pu.visualize_cbfdotlatentunbiased(next_obs, self.cbfd,
                             file,
                             env=self.env,coeff=coeff)

    def plotlatentgroundtruth(self, file, replay_buffer,replay_buffer_unsafe):
        #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']
        if replay_buffer_unsafe!=None:
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
            if self.params['mean']=='meancbf':
                #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                if self.params['boundary']=='no':
                    out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                elif self.params['boundary']=='yes':
                    out_dictus = replay_buffer_unsafe.sample_boundary_meancbf(self.batchsize,'hvn')#(self.params['cbfd_batch_size'])#256
            else:
                out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            next_obsus = out_dictus['next_obs']#rdo = out_dict['rdo']
            next_obs=np.vstack((next_obs,next_obsus))
        #next_obs = out_dict['next_obs_relative']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdotlatentgroundtruth(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plotlatent(self, file, replay_buffer,replay_buffer_unsafe):
        #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']
        if replay_buffer_unsafe!=None:
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
            if self.params['mean']=='meancbf':
                #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                if self.params['boundary']=='no':
                    out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                elif self.params['boundary']=='yes':
                    out_dictus = replay_buffer_unsafe.sample_boundary_meancbf(self.batchsize,'hvn')#(self.params['cbfd_batch_size'])#256
            else:
                out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            next_obsus = out_dictus['next_obs']#rdo = out_dict['rdo']
            next_obs=np.vstack((next_obs,next_obsus))
        #next_obs = out_dict['next_obs_relative']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdotlatent(next_obs, self.cbfd,
                             file,
                             env=self.env)
