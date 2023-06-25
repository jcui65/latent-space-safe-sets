
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
        if self.unsafebuffer=='yes2' or self.unsafebuffer=='yesm2':
            self.batchsize=int(self.params['cbfd_batch_size']/2)#
            #log.info('self.batchsize hope 128:%d'% (self.batchsize))#it is 128!
        else:
            self.batchsize=self.params['cbfd_batch_size']#int(self.params['cbfd_batch_size']/2)#
            #log.info('self.batchsize hope 256:%d'% (self.batchsize))
        self.batchsize0s=64
        self.batchsize0to9=32
        #self.batchsize0=32
        #self.batchsize1=32
        #self.batchsize2=32
        #self.batchsize3=32
        #self.batchsize4=32
        #self.batchsize5=32
        #self.batchsize6=32
        #self.batchsize7=32
        #self.batchsize8=32
        #self.batchsize9=32
        self.batchsize10=128
        self.batchsize0so=64#0 success online
        self.batchsize10o=128#64#10 (1.0) online
        self.env_name = params['env']
        self.gammasafe=params['gammasafe']
        self.gammaunsafe=params['gammaunsafe']

    def initial_train(self, replay_buffer, update_dir,replay_buffer_unsafe=None):
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
            if self.params['env']=='push' and self.params['push_cbf_strategy']==2:
                rdo,rdn, hvo,hvn, hvd = out_dict['rdoef'], out_dict['rdnef'],out_dict['hvoef'],out_dict['hvnef'], out_dict['hvdef']  # 0 or 1
            else:
                rdo,rdn, hvo,hvn, hvd = out_dict['rdo'], out_dict['rdn'],out_dict['hvo'],out_dict['hvn'], out_dict['hvd']  # 0 or 1
            #print('hvn.shape',hvn.shape)
            hvo=hvo-self.params['rectify']
            hvn=hvn-self.params['rectify']
            #print('hvn',hvn)#sanity check passed!#
            #log.info(hvn)
            if replay_buffer_unsafe!=None:
                #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
                if self.params['mean']=='meancbf':
                    out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    #log.info('training the mean version of the CBF!')#sanity check passed!
                else:
                    out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
                obsus=out_dictus['obs']#us means unsafe
                rdous,rdnus, hvous,hvnus, hvdus = out_dictus['rdo'], out_dictus['rdn'],out_dictus['hvo'],out_dictus['hvn'], out_dictus['hvd']  # 0 or 1
                hvous=hvous-self.params['rectify']#thus, the rectify should be 0.05 not -0.05
                hvnus=hvnus-self.params['rectify']
                #print('hvnus',hvnus)#sanity check passed!#
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

    def update(self, replay_buffer, update_dir,replay_buffer_unsafe=None):
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
                #hvo=hvo-self.params['rectify']
                hvn=hvn-self.params['rectify']
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
                    hvous=hvous-self.params['rectify']
                    hvnus=hvnus-self.params['rectify']#already rectified!
                    obs=np.vstack((obs,obsus))
                    #print('obsnew.shape',obs.shape)(256,32)
                    #print('hvnold.shape',hvn.shape)
                    hvn=np.concatenate((hvn,hvnus))
                    #print('hvnnew.shape',hvn.shape)
                    shuffleind=np.random.permutation(obs.shape[0])
                    obs=obs[shuffleind]
                    hvn=hvn[shuffleind]
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
                    deal=min(dhzepochave,1.5*self.params['dhz'])#will it work as expected?deal for dhz epoch ave legit
                else:#not decreasing the dhz!#1.5 not too big nor too small!
                    deal=min(dhzepochave,1.5*self.params['dhz'])#will it work as expected?deal for dhz epoch ave legit
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

    def initial_train_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
        if self.cbfd.trained:
            self.plot(os.path.join(update_dir, "cbfd_start.pdf"), replay_buffer_success,replay_buffer_unsafe)
            self.plotlatent(os.path.join(update_dir, "cbfdlatent_start.pdf"), replay_buffer_success,replay_buffer_unsafe)
            return

        log.info('Beginning cbfdot initial optimization')
        #this ground truth only work for milestone1
        #self.plotlatentgroundtruth(os.path.join(update_dir, "cbfdgroundtruth.pdf"), replay_buffer,replay_buffer_unsafe)#if not spb, then don't plot
        for i in range(self.params['cbfd_init_iters']):#10000
            #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
            if self.params['mean']=='meancbf':#all offline success trajectory
                out_dict = replay_buffer_success.samplemeancbf(self.batchsize)#sanity check passed!#(self.params['cbfd_batch_size'])#256
                #log.info('training the mean version of the CBF!')
            else:#all offline success trajectory
                out_dict = replay_buffer_success.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            obs,next_obs,constr=out_dict['obs'],out_dict['next_obs'],out_dict['constraint']
            if self.params['ways']==1:#not to change the directory/data!
                constr=np.where(constr<0.99,0,constr)
            cbfv=constr*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
            #print('cbfv',cbfv)#check passed!#
            #print('obs.shape',obs.shape)#(128,32)
            loss, info = self.cbfd.update_m2s(obs,next_obs, cbfv, already_embedded=True)  #
            #print(loss)
            #print(info)
            self.loss_plotter.add_data(info)#self.constr.update, not self.update!
            #if replay_buffer_unsafe!=None:#Now replay_buffe_unsafe is not optional, it is now required!
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
            #if self.params['mean']=='meancbf':
                #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                #log.info('training the mean version of the CBF!')#sanity check passed!
            #else:
                #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            if self.params['mean']=='meancbf':
                out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                if self.params['boundary']=='yes':
                    out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256
            else:#all are offline trajectories, but not necessarily violation
                out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
                if self.params['boundary']=='yes':#all trajectories are offline violation
                    out_dictusb = replay_buffer_unsafe.sample_boundary_m2(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256

            obsus,next_obsus,construs=out_dictus['obs'],out_dictus['next_obs'],out_dictus['constraint']
            #print('obsus.shape',obsus.shape)#(128,32)#sanity check passed!
            if self.params['ways']==1:#not to change the directory/data!
                #print('oldconstrus',construs)#sanity check passed!#
                construs=np.where(construs<0.99,0,construs)
                #print('01construs',construs)#sanity check passed!#
            cbfvus=construs*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
            #print('cbfvus',cbfvus)#check passed!#
            loss, info = self.cbfd.update_m2u(obsus,next_obsus, cbfvus, already_embedded=True)  #info is a dictionary
            self.loss_plotter.add_data(info)
            if self.params['boundary']=='yes':
                obsusb,next_obsusb,construsb=out_dictusb['obs'],out_dictusb['next_obs'],out_dictusb['constraint']
                #print('obsusb.shape',obsusb.shape)#(128,32)#sanity check passed!
                if self.params['ways']==1:#not to change the directory/data!
                    construsb=np.where(construsb<0.99,0,construsb)
                cbfvusb=construsb*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                #print('cbfvusb',cbfvusb)#check passed!#
                loss, info = self.cbfd.update_m2u(obsusb,next_obsusb, cbfvusb, already_embedded=True)  #info is a dictionary
                self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating cbfdot function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "cbfd%d.pdf" % i), replay_buffer_success,replay_buffer_unsafe)
                #self.plotlatent(os.path.join(update_dir, "cbfdlatent%d.pdf" % i), replay_buffer,replay_buffer_unsafe)#nothing is plotted if not spb
                #self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased%d-11.pdf" % i), replay_buffer,replay_buffer_unsafe,
                                        #coeff=1)  # a few lines later
                #self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased%d-13.pdf" % i), replay_buffer,replay_buffer_unsafe,coeff=1/3)  # a few lines later
                #self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased%d-14.pdf" % i), replay_buffer,replay_buffer_unsafe,
                                        #coeff=1 / 4)  # a few lines later
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.cbfd.save(os.path.join(update_dir, 'cbfd_%d.pth' % i))

        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def initial_train_m2_0109(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
        if self.cbfd.trained:
            self.plot0109(os.path.join(update_dir, "cbfd_start.pdf"), replay_buffer_success,replay_buffer_unsafe)
            self.plotlatent(os.path.join(update_dir, "cbfdlatent_start.pdf"), replay_buffer_success,replay_buffer_unsafe)
            return

        log.info('Beginning cbfdot initial optimization')
        #this ground truth only work for milestone1
        #self.plotlatentgroundtruth(os.path.join(update_dir, "cbfdgroundtruth.pdf"), replay_buffer,replay_buffer_unsafe)#if not spb, then don't plot
        for i in range(self.params['cbfd_init_iters']):#10000
            #print('i',i)
            #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
            if self.params['mean']=='meancbf':#all offline success trajectory
                out_dict = replay_buffer_success.samplemeancbf(self.batchsize0s)#sanity check passed!#(self.params['cbfd_batch_size'])#256
                #log.info('training the mean version of the CBF!')
            else:#all offline success trajectory
                out_dict = replay_buffer_success.sample(self.batchsize0s)#(self.params['cbfd_batch_size'])#256
            obs,next_obs,constr=out_dict['obs'],out_dict['next_obs'],out_dict['constraint']#focus on expanding the safe zone?
            #cbfv=constr*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
            #print('cbfv',cbfv)#check passed!#
            #print('obs.shape',obs.shape)#(128,32)
            loss, info = self.cbfd.update_m2s_0109(obs,next_obs, constr, already_embedded=True)#use the constraint information by yourself!  #
            #print('losssafe',loss)
            #print(loss)
            #print(info)
            self.loss_plotter.add_data(info)#self.constr.update, not self.update!
            #if replay_buffer_unsafe!=None:#Now replay_buffe_unsafe is not optional, it is now required!
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
            #if self.params['mean']=='meancbf':
                #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                #log.info('training the mean version of the CBF!')#sanity check passed!
            #else:
                #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            #add a for loop to sample from 0,0.1 to 1.0!
            if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
                out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(self.batchsize10,'constraint',1)#(self.params['cbfd_batch_size'])#256
            else:#all are offline trajectories, but not necessarily violation
                out_dictusb = replay_buffer_unsafe.sample_boundary_m2_0109(self.batchsize10,'constraint',1)#(self.params['cbfd_batch_size'])#256
            
            obsusb,next_obsusb,construsb=out_dictusb['obs'],out_dictusb['next_obs'],out_dictusb['constraint']#it should be 1#
            #print(obsusb.shape,construsb.shape)#(128,32),(128,)
            #altogether there will be 448 samples!
            ten=int(self.params['stepstohell'])
            #print('reach this stage0!')#
            for j in range(ten):
                if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
                    out_dictusbi = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(self.batchsize0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
                else:#all are offline trajectories, but not necessarily violation
                    out_dictusbi = replay_buffer_unsafe.sample_boundary_m2_0109(self.batchsize0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
                obsusbi,next_obsusbi,construsbi=out_dictusbi['obs'],out_dictusbi['next_obs'],out_dictusbi['constraint']
                #print(obsusbi.shape,construsbi.shape)
                obsusb=np.vstack((obsusb,obsusbi))
                next_obsusb=np.vstack((next_obsusb,next_obsusbi))
                construsb=np.concatenate((construsb,construsbi))
            shuffleind=np.random.permutation(obsusb.shape[0])
            obsusb=obsusb[shuffleind]
            #print('obsusb.shape',obsusb.shape)
            next_obsusb=next_obsusb[shuffleind]
            construsb=construsb[shuffleind]
            #print('obsusb.shape',obsusb.shape)#(128,32)#sanity check passed!
            #cbfvusb=construsb*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
            #print('cbfvusb',cbfvusb)#check passed!#
            #print('reach this stage1!')#
            loss, info = self.cbfd.update_m2u_0109(obsusb,next_obsusb, construsb, already_embedded=True)  #info is a dictionary
            #print('lossunsafe',loss)
            #print('reach this stage2!')#
            self.loss_plotter.add_data(info)#push back from possibly too optimistic safe zone?

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating cbfdot function heatmap')
                self.loss_plotter.plot()
                self.plot0109(os.path.join(update_dir, "cbfd%d.pdf" % i), replay_buffer_success,replay_buffer_unsafe)
                #self.plot0109safes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success,self.batchsize0s,'soff')#s means safe, off means offline#this is using plan a
                #self.plot0109safes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success_online,self.batchsize0so,'son')#this is using plan a
                #for k in range(10):
                    #self.plot0109unsafes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_unsafe,k/10,self.batchsize0to9,'us'+str(k))#this is using plan a
                #self.plot0109unsafes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_unsafe,1,self.batchsize10,'us10')#us means unsafe
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.cbfd.save(os.path.join(update_dir, 'cbfd_%d.pth' % i))

        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def update_m2(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
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
                if self.params['mean']=='meancbf':#online success trajectory! which may contain no random good and random good!
                    out_dict = replay_buffer_success.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                else:
                    out_dict = replay_buffer_success.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
                obs,next_obs,constr=out_dict['obs'],out_dict['next_obs'],out_dict['constraint']
                if self.params['ways']==1:#not to change the directory/data!
                    constr=np.where(constr<0.99,0,constr)
                cbfv=constr*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                loss, info = self.cbfd.update_m2s(obs, next_obs,cbfv, already_embedded=True)  #info is a dictionary
                self.loss_plotter.add_data(info)
                #if replay_buffer_unsafe!=None:#it is now required!
                #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
                if self.params['mean']=='meancbf':
                    #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    #if self.params['boundary']=='no':#all unsafe trajectories, offline and online! some violate constraints!
                    out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    if self.params['boundary']=='yes':
                        out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256
                else:
                    out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    if self.params['boundary']=='yes':#all unsafe trajectories, all of them are constraint violating!
                        out_dictusb = replay_buffer_unsafe.sample_boundary_m2(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256
                #obsus=out_dictus['obs']#us means unsafe
                #print('obsus.shape',obsus.shape)(128,32)
                obsus,next_obsus,construs=out_dictus['obs'],out_dictus['next_obs'],out_dictus['constraint']
                if self.params['ways']==1:#not to change the directory/data!
                    construs=np.where(construs<0.99,0,construs)
                cbfvus=construs*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                loss, info = self.cbfd.update_m2u(obsus,next_obsus, cbfvus, already_embedded=True)  #info is a dictionary
                self.loss_plotter.add_data(info)

                obsusb,next_obsusb,construsb=out_dictusb['obs'],out_dictusb['next_obs'],out_dictusb['constraint']
                if self.params['ways']==1:#not to change the directory/data!
                    construsb=np.where(construsb<0.99,0,construsb)
                cbfvusb=construsb*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                loss, info = self.cbfd.update_m2u(obsusb,next_obsusb, cbfvusb, already_embedded=True)  #info is a dictionary
                self.loss_plotter.add_data(info)
                #cbfloss=info['cbf_total']#this is the real cbf loss
                loss1=info['old_safe']#notice that it is multiplied by w1!
                loss2=info['new_safe']#notice that it is multiplied by w2(=w1)!
                loss6=info['old_unsafe']#notice that it is multiplied by w1!
                loss7=info['new_unsafe']#notice that it is multiplied by w2(=w1)!
                cbfloss=(loss6/self.params['w6']+loss7/self.params['w7'])/2#happens to be the right choice!
                '''
                if self.env_name=='reacher':
                    dhzepochave+=np.sqrt(cbfloss)#faithfully record it!#np.sqrt(min(loss,10))#over 10 is too crazy!
                elif self.env_name=='push':
                    dhzepochave+=np.sqrt(cbfloss)#
                elif self.env_name=='spb':
                '''
                dhzepochave+=cbfloss#np.sqrt(cbfloss)##print('just hold it now!')
            dhzepochave=dhzepochave/self.params['cbfd_update_iters']
            if dhzepochave<1e-15:#to avoid any numerical issues!
                dhzepochave=0#dhzepochave#already done with the processing!#/100#this 100 is because of 10000^0.5=100
            log.info('the average dhz of this epochs: %f'%(dhzepochave))
            #if self.params['dynamic_dhz']=='yes':
                #if self.env_name=='reacher':
                    #deal=min(dhzepochave,1.5*self.params['dhz'])#will it work as expected?deal for dhz epoch ave legit
                #else:#not decreasing the dhz!#1.5 not too big nor too small!
                #deal=min(dhzepochave,1.5*self.params['dhz'])#will it work as expected?deal for dhz epoch ave legit
            #else:
            deal=dhzepochave#no need to make the above if statement!
            log.info('Creating cbf dot function heatmap')
            self.loss_plotter.plot()
            self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success,replay_buffer_unsafe)#this is using plan a
            #self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
            #self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased.pdf" ), replay_buffer, coeff=1)
            #self.plotconly(os.path.join(update_dir, "cbfdcircle.pdf"), replay_buffer)  # a few lines later
            #self.plotlatent(os.path.join(update_dir, "cbfdlatent.pdf"), replay_buffer_success,replay_buffer_unsafe)
            #self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased-11.pdf"), replay_buffer_success,replay_buffer_unsafe,
                                    #coeff=1)  # a few lines later
            #self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased-13.pdf"), replay_buffer_success,replay_buffer_unsafe,coeff=1/3)  # a few lines later
            #self.plotlatentunbiased(os.path.join(update_dir, "cbfdlatentunbiased-14.pdf"), replay_buffer_success,replay_buffer_unsafe,
                                    #coeff=1 / 4)  # a few lines later
            #self.plotlatentgroundtruth(os.path.join(update_dir, "cbfdgroundtruth.pdf"), replay_buffer)
            self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
            return deal

    def update_m2_0109(self, replay_buffer_success, update_dir,replay_buffer_unsafe):
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
                if self.params['mean']=='meancbf':#all offline success trajectory
                    out_dict = replay_buffer_success.samplemeancbf(self.batchsize0s)#sanity check passed!#(self.params['cbfd_batch_size'])#256
                    #log.info('training the mean version of the CBF!')
                else:#all offline success trajectory
                    out_dict = replay_buffer_success.sample(self.batchsize0s)#(self.params['cbfd_batch_size'])#256
                obs,next_obs,constr=out_dict['obs'],out_dict['next_obs'],out_dict['constraint']#focus on expanding the safe zone?
                #cbfv=constr*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                #print('cbfv',cbfv)#check passed!#
                #print('obs.shape',obs.shape)#(128,32)
                loss, info = self.cbfd.update_m2s_0109(obs,next_obs, constr, already_embedded=True)#use the constraint information by yourself!  #
                #print('losssafe',loss)
                #print(loss)
                #print(info)
                self.loss_plotter.add_data(info)#self.constr.update, not self.update!
                #if replay_buffer_unsafe!=None:#Now replay_buffe_unsafe is not optional, it is now required!
                #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)#256
                #if self.params['mean']=='meancbf':
                    #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    #log.info('training the mean version of the CBF!')#sanity check passed!
                #else:
                    #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
                #add a for loop to sample from 0,0.1 to 1.0!
                if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
                    out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(self.batchsize10,'constraint',1)#(self.params['cbfd_batch_size'])#256
                else:#all are offline trajectories, but not necessarily violation
                    out_dictusb = replay_buffer_unsafe.sample_boundary_m2_0109(self.batchsize10,'constraint',1)#(self.params['cbfd_batch_size'])#256
                
                obsusb,next_obsusb,construsb=out_dictusb['obs'],out_dictusb['next_obs'],out_dictusb['constraint']#it should be 1#
                #print(obsusb.shape,construsb.shape)#(128,32),(128,)
                #altogether there will be 448 samples!
                ten=int(self.params['stepstohell'])
                #print('reach this stage0!')#
                for j in range(ten):
                    if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
                        out_dictusbi = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(self.batchsize0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
                    else:#all are offline trajectories, but not necessarily violation
                        out_dictusbi = replay_buffer_unsafe.sample_boundary_m2_0109(self.batchsize0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
                    obsusbi,next_obsusbi,construsbi=out_dictusbi['obs'],out_dictusbi['next_obs'],out_dictusbi['constraint']
                    #print(obsusbi.shape,construsbi.shape)
                    obsusb=np.vstack((obsusb,obsusbi))#pay attention to the dimension!
                    next_obsusb=np.vstack((next_obsusb,next_obsusbi))
                    construsb=np.concatenate((construsb,construsbi))
                shuffleind=np.random.permutation(obsusb.shape[0])
                obsusb=obsusb[shuffleind]
                #print('obsusb.shape',obsusb.shape)
                next_obsusb=next_obsusb[shuffleind]
                construsb=construsb[shuffleind]
                #print('obsusb.shape',obsusb.shape)#(128,32)#sanity check passed!
                #cbfvusb=construsb*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                #print('cbfvusb',cbfvusb)#check passed!#
                #print('reach this stage1!')#
                loss, info = self.cbfd.update_m2u_0109(obsusb,next_obsusb, construsb, already_embedded=True)  #info is a dictionary
                #print('lossunsafe',loss)
                #print('reach this stage2!')#
                self.loss_plotter.add_data(info)#push back from possibly too optimistic safe zone?

                #cbfloss=info['cbf_total']#this is the real cbf loss
                loss1=info['old_safe']#notice that it is multiplied by w1!
                loss2=info['new_safe']#notice that it is multiplied by w2(=w1)!
                loss6=info['old_unsafe']#notice that it is multiplied by w1!
                loss7=info['new_unsafe']#notice that it is multiplied by w2(=w1)!
                cbfloss=(loss6/self.params['w6']+loss7/self.params['w7'])/2#happens to be the right choice!
                '''
                if self.env_name=='reacher':
                    dhzepochave+=np.sqrt(cbfloss)#faithfully record it!#np.sqrt(min(loss,10))#over 10 is too crazy!
                elif self.env_name=='push':
                    dhzepochave+=np.sqrt(cbfloss)#
                elif self.env_name=='spb':
                '''
                dhzepochave+=cbfloss#np.sqrt(cbfloss)##print('just hold it now!')
            dhzepochave=dhzepochave/self.params['cbfd_update_iters']
            if dhzepochave<1e-15:#to avoid any numerical issues!
                dhzepochave=0#dhzepochave#already done with the processing!#/100#this 100 is because of 10000^0.5=100
            log.info('the average dhz of this epochs: %f'%(dhzepochave))

            deal=dhzepochave#no need to make the above if statement!
            log.info('Creating cbf dot function heatmap')
            self.loss_plotter.plot()
            self.plot0109(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success,replay_buffer_unsafe)#this is using plan a
            self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
            return deal

    def update_m2_withonline(self, replay_buffer_success, update_dir,replay_buffer_unsafe, replay_buffer_success_online,replay_buffer_unsafe_online):
        if self.params['train_cbf']=='no':
            log.info('No episodic cbf dot update optimization!')
        else:
            if self.params['train_cbf']=='no2':
                self.params['cbfd_lr']=0
                log.info('No episodic cbf dot update optimization but show loss on new data!')
            else:
                log.info('Beginning cbf dot update optimization including an online buffer!')
            dhzepochave=0
            #log.info('cbfd_lr: %f'%(self.params['cbfd_lr']))
            lens=len(replay_buffer_success)
            lenu=len(replay_buffer_unsafe)
            lenso=len(replay_buffer_success_online)
            lenuo=len(replay_buffer_unsafe_online)
            #lentotal=lens+lenu+lenso+lenuo
            lentotals=lens+lenso
            lentotalu=lenu+lenuo
            ratios=lens/lentotals
            ratioso=lenso/lentotals
            ratiou=lenu/lentotalu
            ratiouo=lenuo/lentotalu
            #k=0
            for _ in trange(self.params['cbfd_update_iters']):#512
                if self.params['mean']=='meancbf':#all offline success trajectory
                    out_dict = replay_buffer_success.samplemeancbf(self.batchsize0s)#sanity check passed!#(self.params['cbfd_batch_size'])#256
                    #log.info('training the mean version of the CBF!')
                else:#all offline success trajectory
                    out_dict = replay_buffer_success.sample(self.batchsize0s)#(self.params['cbfd_batch_size'])#256
                if self.params['mean']=='meancbf':#all offline success trajectory
                    out_dicto = replay_buffer_success_online.samplemeancbf(self.batchsize0so)#sanity check passed!#(self.params['cbfd_batch_size'])#256
                    #log.info('training the mean version of the CBF!')
                else:#all offline success trajectory
                    out_dicto = replay_buffer_success_online.sample(self.batchsize0so)#(self.params['cbfd_batch_size'])#256
                obs,next_obs,constr=out_dict['obs'],out_dict['next_obs'],out_dict['constraint']#focus on expanding the safe zone?
                obso,next_obso,constro=out_dicto['obs'],out_dicto['next_obs'],out_dicto['constraint']#focus on expanding the safe zone?
                
                if self.params['ways']==1:#not to change the directory/data!
                    constr=np.where(constr<0.99,0,constr)
                    constro=np.where(constro<0.99,0,constro)
                cbfv=constr*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                cbfvo=constro*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                loss, info = self.cbfd.update_m2s(obs,next_obs, cbfv, already_embedded=True)#use the constraint information by yourself!  #
                self.loss_plotter.add_data(info)#self.constr.update, not self.update!
                loss, info = self.cbfd.update_m2s_online(obso,next_obso, cbfvo, already_embedded=True)#use the constraint information by yourself!  #
                self.loss_plotter.add_data(info)#self.constr.update, not self.update!
                #if k==0:
                    #log.info('online success buffer has been used as expected!')
                
                if ratiouo>0:#this means online unsafe incident happens!
                    unsafeobatch=max(2,int(ratiouo*self.batchsize))
                    unsafebatch=self.batchsize-unsafeobatch
                else:
                    unsafebatch=self.batchsize#128
                    unsafeobatch=0
                    #if k==0:
                        #log.info('no online safety violations!')
                
                if self.params['mean']=='meancbf':
                    #out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    #if self.params['boundary']=='no':#all unsafe trajectories, offline and online! some violate constraints!
                    out_dictus = replay_buffer_unsafe.samplemeancbf(unsafebatch)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    if self.params['boundary']=='yes':
                        out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2(unsafebatch,'constraint')#(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256
                else:
                    out_dictus = replay_buffer_unsafe.sample(unsafebatch)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
                    if self.params['boundary']=='yes':#all unsafe trajectories, all of them are constraint violating!
                        out_dictusb = replay_buffer_unsafe.sample_boundary_m2(unsafebatch,'constraint')#(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256
                #obsus=out_dictus['obs']#us means unsafe
                #print('obsus.shape',obsus.shape)(128,32)
                obsus,next_obsus,construs=out_dictus['obs'],out_dictus['next_obs'],out_dictus['constraint']
                obsusb,next_obsusb,construsb=out_dictusb['obs'],out_dictusb['next_obs'],out_dictusb['constraint']#it should be 1#

                if unsafeobatch>=1:#ratiouo>0:#
                    if self.params['mean']=='meancbf':
                        out_dictuso = replay_buffer_unsafe_online.samplemeancbf(unsafeobatch)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
                        if self.params['boundary']=='yes':
                            out_dictusbo = replay_buffer_unsafe_online.sample_boundary_meancbf_m2(unsafeobatch,'constraint')#(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256
                    else:
                        out_dictuso = replay_buffer_unsafe_online.sample(unsafeobatch)#(self.batchsize)#(self.params['cbfd_batch_size'])#256
                        if self.params['boundary']=='yes':#all unsafe trajectories, all of them are constraint violating!
                            out_dictusbo = replay_buffer_unsafe_online.sample_boundary_m2(unsafeobatch,'constraint')#(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256
                    #obsus=out_dictus['obs']#us means unsafe#print('obsus.shape',obsus.shape)(128,32)
                    obsuso,next_obsuso,construso=out_dictuso['obs'],out_dictuso['next_obs'],out_dictuso['constraint']
                    obsusbo,next_obsusbo,construsbo=out_dictusbo['obs'],out_dictusbo['next_obs'],out_dictusbo['constraint']#it should be 1#
                    obsus=np.vstack((obsus,obsuso))#pay attention to the dimension!
                    next_obsus=np.vstack((next_obsus,next_obsuso))
                    construs=np.concatenate((construs,construso))
                    obsusb=np.vstack((obsusb,obsusbo))#pay attention to the dimension!
                    next_obsusb=np.vstack((next_obsusb,next_obsusbo))
                    construsb=np.concatenate((construsb,construsbo))
                    '''
                    if self.params['ways']==1:#not to change the directory/data!
                        construso=np.where(construso<0.99,0,construso)
                        construsbo=np.where(construsbo<0.99,0,construsbo)
                    cbfvuso=construso*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                    cbfvusbo=construsbo*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                    
                    loss, info = self.cbfd.update_m2u(obsuso,next_obsuso, cbfvuso, already_embedded=True)  #info is a dictionary
                    self.loss_plotter.add_data(info)
                    loss, info = self.cbfd.update_m2u(obsusbo,next_obsusbo, cbfvusbo, already_embedded=True)  #info is a dictionary
                    self.loss_plotter.add_data(info)#push back from possibly too optimistic safe zone?
                    #if k==0:
                        #log.info('online violation has been taken into account!')
                    '''
                if self.params['ways']==1:#not to change the directory/data!
                    construs=np.where(construs<0.99,0,construs)
                    construsb=np.where(construsb<0.99,0,construsb)
                cbfvus=construs*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                cbfvusb=construsb*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
                
                loss, info = self.cbfd.update_m2u(obsus,next_obsus, cbfvus, already_embedded=True)  #info is a dictionary
                self.loss_plotter.add_data(info)
                loss, info = self.cbfd.update_m2u(obsusb,next_obsusb, cbfvusb, already_embedded=True)  #info is a dictionary
                self.loss_plotter.add_data(info)#push back from possibly too optimistic safe zone?

                #needs further change!
                #cbfloss=info['cbf_total']#this is the real cbf loss
                loss1=info['old_safe']#notice that it is multiplied by w1!
                loss2=info['new_safe']#notice that it is multiplied by w2(=w1)!
                loss6=info['old_unsafe']#notice that it is multiplied by w1!
                loss7=info['new_unsafe']#notice that it is multiplied by w2(=w1)!
                cbfloss=(loss6/self.params['w6']+loss7/self.params['w7'])/2#happens to be the right choice!
                dhzepochave+=cbfloss#np.sqrt(cbfloss)##print('just hold it now!')
                #k+=1
            dhzepochave=dhzepochave/self.params['cbfd_update_iters']
            if dhzepochave<1e-15:#to avoid any numerical issues!
                dhzepochave=0#dhzepochave#already done with the processing!#/100#this 100 is because of 10000^0.5=100
            log.info('the average dhz of this epochs: %f'%(dhzepochave))

            deal=dhzepochave#no need to make the above if statement!
            log.info('Creating cbf dot function heatmap')
            self.loss_plotter.plot()
            #self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success,replay_buffer_unsafe)#this is using plan a
            #self.plot0109(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success,replay_buffer_unsafe,replay_buffer_success_online)#this is using plan a
            self.plot0109safes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success,self.batchsize0s,'soff')#s means safe, off means offline#this is using plan a
            self.plot0109safes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success_online,self.batchsize0so,'son')#this is using plan a
            for k in range(10):
                self.plot0109unsafes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_unsafe,k/10,self.batchsize0to9,'us'+str(k))#this is using plan a
            self.plot0109unsafes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_unsafe,1,self.batchsize10,'us10')#us means unsafe
            
            self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
            return deal

    def update_m2_0109_withonline(self, replay_buffer_success, update_dir,replay_buffer_unsafe, replay_buffer_success_online,replay_buffer_unsafe_online):
        if self.params['train_cbf']=='no':
            log.info('No episodic cbf dot update optimization!')
        else:
            if self.params['train_cbf']=='no2':
                self.params['cbfd_lr']=0
                log.info('No episodic cbf dot update optimization but show loss on new data!')
            else:
                log.info('Beginning cbf dot update optimization including an online buffer!')
            dhzepochave=0
            lens=len(replay_buffer_success)
            lenu=len(replay_buffer_unsafe)
            lenso=len(replay_buffer_success_online)
            lenuo=len(replay_buffer_unsafe_online)
            #lentotal=lens+lenu+lenso+lenuo
            lentotals=lens+lenso
            lentotalu=lenu+lenuo
            ratios=lens/lentotals
            ratioso=lenso/lentotals
            ratiou=lenu/lentotalu
            ratiouo=lenuo/lentotalu
            #log.info('cbfd_lr: %f'%(self.params['cbfd_lr']))
            k=0
            for _ in trange(self.params['cbfd_update_iters']):#512
                if self.params['mean']=='meancbf':#all offline success trajectory
                    out_dict = replay_buffer_success.samplemeancbf(self.batchsize0s)#sanity check passed!#(self.params['cbfd_batch_size'])#256
                    #log.info('training the mean version of the CBF!')
                else:#all offline success trajectory
                    out_dict = replay_buffer_success.sample(self.batchsize0s)#(self.params['cbfd_batch_size'])#256
                obs,next_obs,constr=out_dict['obs'],out_dict['next_obs'],out_dict['constraint']#focus on expanding the safe zone?
                loss, info = self.cbfd.update_m2s_0109(obs,next_obs, constr, already_embedded=True)#use the constraint information by yourself!  #
                self.loss_plotter.add_data(info)#self.constr.update, not self.update!
                if self.params['mean']=='meancbf':#all offline success trajectory
                    out_dicto = replay_buffer_success_online.samplemeancbf(self.batchsize0so)#sanity check passed!#(self.params['cbfd_batch_size'])#256
                    #log.info('training the mean version of the CBF!')
                else:#all offline success trajectory
                    out_dicto = replay_buffer_success_online.sample(self.batchsize0so)#(self.params['cbfd_batch_size'])#256
                obso,next_obso,constro=out_dicto['obs'],out_dicto['next_obs'],out_dicto['constraint']#focus on expanding the safe zone?
                loss, info = self.cbfd.update_m2s_0109_online(obso,next_obso, constro, already_embedded=True)#use the constraint information by yourself!  #
                self.loss_plotter.add_data(info)#self.constr.update, not self.update!
                #if k==0:
                    #log.info('online success buffer has been used as expected!')
                
                if ratiouo>0:#this means online unsafe incident happens!
                    unsafeobatch10=max(2,int(ratiouo*self.batchsize10))#max(2,int(ratiouo*self.batchsize))#
                    unsafebatch10=self.batchsize10-unsafeobatch10#self.batchsize-unsafeobatch10
                    unsafeobatch0to9=max(1,int(ratiouo*self.batchsize0to9))
                    unsafebatch0to9=self.batchsize0to9-unsafeobatch0to9
                else:
                    unsafebatch10=self.batchsize10#self.batchsize#128
                    unsafeobatch10=0
                    unsafeobatch0to9=0
                    unsafebatch0to9=self.batchsize0to9
                
                if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
                    #out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(self.batchsize10,'constraint',1)#(self.params['cbfd_batch_size'])#256
                    out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(unsafebatch10,'constraint',1)#(self.params['cbfd_batch_size'])#256
                else:#all are offline trajectories, but not necessarily violation
                    #out_dictusb = replay_buffer_unsafe.sample_boundary_m2_0109(self.batchsize10,'constraint',1)#(self.params['cbfd_batch_size'])#256
                    out_dictusb = replay_buffer_unsafe.sample_boundary_m2_0109(unsafebatch10,'constraint',1)#(self.params['cbfd_batch_size'])#256
                obsusb,next_obsusb,construsb=out_dictusb['obs'],out_dictusb['next_obs'],out_dictusb['constraint']#it should be 1#
                #print(obsusb.shape,construsb.shape)#(128,32),(128,)
                #altogether there will be 448 samples!
                #add a for loop to sample from 0,0.1 to 0.9!
                ten=10#int(self.params['stepstohell'])
                #print('reach this stage0!')#
                for j in range(ten):
                    if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
                        out_dictusbi = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(unsafebatch0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
                    else:#all are offline trajectories, but not necessarily violation
                        out_dictusbi = replay_buffer_unsafe.sample_boundary_m2_0109(unsafebatch0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
                    obsusbi,next_obsusbi,construsbi=out_dictusbi['obs'],out_dictusbi['next_obs'],out_dictusbi['constraint']
                    #print(obsusbi.shape,construsbi.shape)
                    obsusb=np.vstack((obsusb,obsusbi))#pay attention to the dimension!
                    next_obsusb=np.vstack((next_obsusb,next_obsusbi))
                    construsb=np.concatenate((construsb,construsbi))
                
                if unsafeobatch10>=1:#ratiouo>0:#
                    if self.params['mean']=='meancbf':
                        out_dictusbo = replay_buffer_unsafe_online.sample_boundary_meancbf_m2(unsafeobatch10,'constraint')#(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256
                    else:
                        out_dictusbo = replay_buffer_unsafe_online.sample_boundary_m2(unsafeobatch10,'constraint')#(self.batchsize,'constraint')#(self.params['cbfd_batch_size'])#256
                    #obsus=out_dictus['obs']#us means unsafe#print('obsus.shape',obsus.shape)(128,32)
                    obsusbo,next_obsusbo,construsbo=out_dictusbo['obs'],out_dictusbo['next_obs'],out_dictusbo['constraint']#it should be 1#
                    obsusb=np.vstack((obsusb,obsusbo))#pay attention to the dimension!
                    next_obsusb=np.vstack((next_obsusb,next_obsusbo))
                    construsb=np.concatenate((construsb,construsbo))
                
                    for j in range(ten):
                        if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
                            out_dictusbio = replay_buffer_unsafe_online.sample_boundary_meancbf_m2_0109(unsafeobatch0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
                        else:#all are offline trajectories, but not necessarily violation
                            out_dictusbio = replay_buffer_unsafe_online.sample_boundary_m2_0109(unsafeobatch0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
                        obsusbio,next_obsusbio,construsbio=out_dictusbio['obs'],out_dictusbio['next_obs'],out_dictusbio['constraint']
                        #print(obsusbi.shape,construsbi.shape)
                        obsusbio=obsusbio.reshape(unsafeobatch0to9,obsusbio.shape[-1])
                        next_obsusbio=next_obsusbio.reshape(unsafeobatch0to9,next_obsusbio.shape[-1])
                        obsusb=np.vstack((obsusb,obsusbio))#pay attention to the dimension!
                        next_obsusb=np.vstack((next_obsusb,next_obsusbio))
                        construsb=np.concatenate((construsb,construsbio))

                shuffleind=np.random.permutation(obsusb.shape[0])
                obsusb=obsusb[shuffleind]
                #print('obsusb.shape',obsusb.shape)
                next_obsusb=next_obsusb[shuffleind]
                construsb=construsb[shuffleind]
                loss, info = self.cbfd.update_m2u_0109(obsusb,next_obsusb, construsb, already_embedded=True)  #info is a dictionary
                self.loss_plotter.add_data(info)#push back from possibly too optimistic safe zone?

                #cbfloss=info['cbf_total']#this is the real cbf loss
                loss1=info['old_safe']#notice that it is multiplied by w1!
                loss2=info['new_safe']#notice that it is multiplied by w2(=w1)!
                loss6=info['old_unsafe']#notice that it is multiplied by w1!
                loss7=info['new_unsafe']#notice that it is multiplied by w2(=w1)!
                cbfloss=(loss6/self.params['w6']+loss7/self.params['w7'])/2#happens to be the right choice!
                dhzepochave+=cbfloss#np.sqrt(cbfloss)##print('just hold it now!')
                k+=1
            dhzepochave=dhzepochave/self.params['cbfd_update_iters']
            if dhzepochave<1e-15:#to avoid any numerical issues!
                dhzepochave=0#dhzepochave#already done with the processing!#/100#this 100 is because of 10000^0.5=100
            log.info('the average dhz of this epochs: %f'%(dhzepochave))

            deal=dhzepochave#no need to make the above if statement!
            log.info('Creating cbf dot function heatmap')
            self.loss_plotter.plot()
            #self.plot0109(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success,replay_buffer_unsafe,replay_buffer_success_online)#this is using plan a
            self.plot0109safes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success,self.batchsize0s,'soff')#s means safe, off means offline#this is using plan a
            self.plot0109safes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_success_online,self.batchsize0so,'son')#this is using plan a
            for k in range(10):
                self.plot0109unsafes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_unsafe,k/10,self.batchsize0to9,'us'+str(k))#this is using plan a
            self.plot0109unsafes(os.path.join(update_dir, "cbfd.pdf"), replay_buffer_unsafe,1,self.batchsize10,'us10')#us means unsafe
            self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))
            return deal

    def plot(self, file, replay_buffer,replay_buffer_unsafe=None):
        #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        if replay_buffer_unsafe!=None:
            #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
            if self.params['mean']=='meancbf':
                out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2(self.batchsize,'constraint')#
                '''
                if self.params['boundary']=='no':
                    out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
                elif self.params['boundary']=='yes':
                    #out_dictus = replay_buffer_unsafe.sample_boundary_meancbf(self.batchsize,'hvn')#(self.params['cbfd_batch_size'])#256
                    out_dictus = replay_buffer_unsafe.sample_boundary_meancbf_m2(self.batchsize,'constraint')#
                '''    
            else:
                out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
                out_dictusb = replay_buffer_unsafe.sample_boundary_m2(self.batchsize,'constraint')#
            next_obsus = out_dictus['next_obs']#rdo = out_dict['rdo']
            next_obs=np.vstack((next_obs,next_obsus))
            next_obsusb = out_dictusb['next_obs']#rdo = out_dict['rdo']
            next_obs=np.vstack((next_obs,next_obsusb))
        #next_obs = out_dict['next_obs_relative']  # rdo = out_dict['rdo']
        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plot0109(self, file, replay_buffer,replay_buffer_unsafe,replay_buffer_online=None):#unsafe buffer is a must now!
        #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(self.batchsize0s)#(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(self.batchsize0s)#(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        '''
        #if replay_buffer_unsafe!=None:
        #out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':
            out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
            out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2(self.batchsize,'constraint')#
            
            if self.params['boundary']=='no':
                out_dictus = replay_buffer_unsafe.samplemeancbf(self.batchsize)#(self.params['cbfd_batch_size'])#256
            elif self.params['boundary']=='yes':
                #out_dictus = replay_buffer_unsafe.sample_boundary_meancbf(self.batchsize,'hvn')#(self.params['cbfd_batch_size'])#256
                out_dictus = replay_buffer_unsafe.sample_boundary_meancbf_m2(self.batchsize,'constraint')#
                
        else:
            out_dictus = replay_buffer_unsafe.sample(self.batchsize)#(self.params['cbfd_batch_size'])#256
            out_dictusb = replay_buffer_unsafe.sample_boundary_m2(self.batchsize,'constraint')#
        next_obsus = out_dictus['next_obs']#rdo = out_dict['rdo']
        next_obs=np.vstack((next_obs,next_obsus))
        next_obsusb = out_dictusb['next_obs']#rdo = out_dict['rdo']
        next_obs=np.vstack((next_obs,next_obsusb))
        '''

        if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
            out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(self.batchsize10,'constraint',1)#(self.params['cbfd_batch_size'])#256
        else:#all are offline trajectories, but not necessarily violation
            out_dictusb = replay_buffer_unsafe.sample_boundary_m2_0109(self.batchsize10,'constraint',1)#(self.params['cbfd_batch_size'])#256
        obsusb,next_obsusb,construsb=out_dictusb['obs'],out_dictusb['next_obs'],out_dictusb['constraint']#it should be 1#
        next_obs=np.vstack((next_obs,next_obsusb))
        #print(obsusb.shape,construsb.shape)#(128,32),(128,)
        #altogether there will be 448 samples!
        ten=10#int(self.params['stepstohell'])
        #print('reach this stage0!')#
        for j in range(ten):
            if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
                out_dictusbi = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(self.batchsize0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
                #print('%dth completed!'%(j))
            else:#all are offline trajectories, but not necessarily violation
                out_dictusbi = replay_buffer_unsafe.sample_boundary_m2_0109(self.batchsize0to9,'constraint',j/ten)#(self.params['cbfd_batch_size'])#256
            obsusbi,next_obsusbi,construsbi=out_dictusbi['obs'],out_dictusbi['next_obs'],out_dictusbi['constraint']
            next_obs=np.vstack((next_obs,next_obsusbi))
        #next_obs = out_dict['next_obs_relative']  # rdo = out_dict['rdo']#there will now be 512 figures!
        if replay_buffer_online!=None:
            if self.params['mean']=='meancbf':
                out_dicto = replay_buffer_online.samplemeancbf(self.batchsize0so)#(self.params['cbfd_batch_size'])#256
            else:
                out_dicto = replay_buffer_online.sample(self.batchsize0so)#(self.params['cbfd_batch_size'])#256
            next_obso = out_dicto['next_obs']#rdo = out_dict['rdo']
            next_obs=np.vstack((next_obs,next_obso))

        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)

    def plot0109safes(self, file, replay_buffer,bs,token):#s means separate!#this is for success and success_online
        #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':
            out_dict = replay_buffer.samplemeancbf(bs)#(self.params['cbfd_batch_size'])#256
        else:
            out_dict = replay_buffer.sample(bs)#(self.params['cbfd_batch_size'])#256
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        
        pu.visualize_cbfdot(next_obs, self.cbfd,file,env=self.env,token=token)

    def plot0109unsafes(self, file, replay_buffer_unsafe,value,bs,token):#unsafe buffer is a must now!#s means separate
        #out_dict = replay_buffer.sample(self.batchsize)#(self.params['cbfd_batch_size']/2)
        if self.params['mean']=='meancbf':#the boundary option is not needed anymore! Because it must be it! Thanks Dr. Nadia!
            out_dictusb = replay_buffer_unsafe.sample_boundary_meancbf_m2_0109(bs,'constraint',value)#(self.params['cbfd_batch_size'])#256
        else:#all are offline trajectories, but not necessarily violation
            out_dictusb = replay_buffer_unsafe.sample_boundary_m2_0109(bs,'constraint',value)#(self.params['cbfd_batch_size'])#256
        obsusb,next_obsusb,construsb=out_dictusb['obs'],out_dictusb['next_obs'],out_dictusb['constraint']#it should be 1#
        #print(obsusb.shape,construsb.shape)#(128,32),(128,)
        #altogether there will be 448 samples!

        pu.visualize_cbfdot(next_obsusb, self.cbfd,file,env=self.env,token=token)

    def plotconly(self, file, replay_buffer,replay_buffer_unsafe=None):
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


    def plotlatentunbiased(self, file, replay_buffer,replay_buffer_unsafe=None,coeff=1):
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

    def plotlatentgroundtruth(self, file, replay_buffer,replay_buffer_unsafe=None):
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

    def plotlatent(self, file, replay_buffer,replay_buffer_unsafe=None):
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
