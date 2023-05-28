import numpy as np
import latentsafesets.utils.pytorch_utils as ptu
import json
import os
class EncodedReplayBuffer:
    """
    This replay buffer uses numpy to efficiently store arbitrary data. Keys can be whatever,
    but once you push data to the buffer new data must all have the same keys (to keep parallel
    arrays parallel).

    This replay buffer replaces all images with their representation from encoder
    """
    def __init__(self, encoder, size=10000,mean='mean'):
        self.size = size
        self.encoder = encoder
        self.mean=mean
        self.data = {}#finally it becomes a dict where each key's value have size number of values
        self._index = 0
        self._len = 0
        #self.im_keys = ('obs', 'next_obs')
        self.im_keys = ('obs', 'next_obs','obs_relative','next_obs_relative')

    def store_transitions(self, transitions):#transitions is 1 traj having 100 steps
        """
        Stores transitions
        :param transitions: a list of dictionaries encoding transitions. Keys can be anything
        """
        assert transitions[-1]['done'] > 0, "Last transition must be end of trajectory"
        for transition in transitions:#a transition is 1 step#It is a dictionary
            self.store_transition(transition)

    def store_transitions_latent(self, transitions):#transitions is 1 traj having 100 steps
        """
        Stores transitions
        :param transitions: a list of dictionaries encoding transitions. Keys can be anything
        """
        assert transitions[-1]['done'] > 0, "Last transition must be end of trajectory"
        for transition in transitions:#a transition is 1 step#It is a dictionary
            self.store_transition_latent(transition)

    def store_dump_transitions(self,transitions,file,update):#transitions is 1 traj having 100 steps
        """
        Stores transitions
        :param transitions: a list of dictionaries encoding transitions. Keys can be anything
        """
        assert transitions[-1]['done'] > 0, "Last transition must be end of trajectory"
        for transition in transitions:#a transition is 1 step#It is a dictionary
            self.store_transition(transition)
        im_fields = ('obs', 'next_obs')
        #traj_no_ims = [{key: frame[key] if key not in im_fields else key: frame[key].tolist() for key in frame}
                   #for frame in trajectory]#trajectory contains 100 frames#turn images into latent states
        n=update#*10+traj
        with open(os.path.join(file, "%d.json" % n), "w") as f:
            json.dump(traj_no_ims, f)#separate trajectory info from images

    def store_transition(self, transition):#a transition is 1 step#It is a dictionary
        if len(self.data) > 0:#at first it is not like this, from second it is like this
            key_set = set(self.data)#the keys of self.data are the keys of transition!
        else:#at first it is like this#self.data is different from self.transition!
            key_set = set(transition)#you only get the keys of that dictionary! python usage!

        # assert key_set == set(transition), "Expected transition to have keys %s" % key_set

        for key in key_set:#it is a set
            data = self.data.get(key, None)#.get() is to get the value of a key
            #print('transition',transition)
            if key in transition:#I added!
                new_data = np.array(transition[key])#it seems already converts value list to array
                if key in self.im_keys:
                    #print('keyname: ',key)#
                    im = np.array(transition[key])#seems to be the image?
                    im = ptu.torchify(im)
                    new_data_mean, new_data_log_std = self.encoder(im[None] / 255)#is it legit?
                    new_data_mean = new_data_mean.squeeze().detach().cpu().numpy()
                    new_data_log_std = new_data_log_std.squeeze().detach().cpu().numpy()#meancbf still works like this!
                    if self.mean=='mean':
                        new_data_log_std=np.clip(new_data_log_std,a_min=None,a_max=-80)#-80 is really very small!
                    new_data = np.dstack((new_data_mean, new_data_log_std)).squeeze()
                    '''
                    #just for testing!#test passed!
                    new_data_mean2, new_data_log_std2 = self.encoder(im[None] / 255)#is it legit?
                    new_data_mean2 = new_data_mean2.squeeze().detach().cpu().numpy()
                    new_data_log_std2 = new_data_log_std2.squeeze().detach().cpu().numpy()
                    #print('new_data_log_std2: ',new_data_log_std2)#(32,)the ordier is e-5~e-7
                    if self.mean=='mean':#the mean is around 0.1. Thus, in the pushing case, the sampling doesn't matter that much!
                        new_data_log_std2=np.clip(new_data_log_std2,a_min=None,a_max=-80)#-80 is really very small!
                    new_data2 = np.dstack((new_data_mean2, new_data_log_std2)).squeeze()
                    new_datadiff=new_data-new_data2
                    '''
                    #print('new_data2',new_data2)#all the log std is -80 now! passed!
                    #print('new_datadiff',new_datadiff)#0,0 as expected! same seed still 0 in sample mode!
                    #print('new_data_mean.shape',new_data_mean.shape)#(32,)
                    #print('new_data_log_std.shape',new_data_log_std.shape)#(32,)
                    #print('new_data.shape',new_data.shape)#(32,2)
                    #print('new_data_mean',new_data_mean)#(32,)
                    #print('new_data_log_std',new_data_log_std)#(32,)

                if data is None:
                    data = np.zeros((self.size, *new_data.shape))#then fill one by one
                data[self._index] = new_data#now fill one by one#this is the data of this key!
                self.data[key] = data#the value of self.data[key] is a np array#the way to init a value in a dict

        self._index = (self._index + 1) % self.size#no more no less, just 10k pieces of data#a queue like dagger
        self._len = min(self._len + 1, self.size)#a thing saturate at self.size!
        #I think I have understood the above function!

    def store_transition_latent(self, transition):#a transition is 1 step#It is a dictionary
        if len(self.data) > 0:#at first it is not like this, from second it is like this
            key_set = set(self.data)#the keys of self.data are the keys of transition!
        else:#at first it is like this#self.data is different from self.transition!
            key_set = set(transition)#you only get the keys of that dictionary! python usage!

        # assert key_set == set(transition), "Expected transition to have keys %s" % key_set

        for key in key_set:#it is a set
            data = self.data.get(key, None)#.get() is to get the value of a key
            #print('transition',transition)
            if key in transition:#I added!
                new_data = np.array(transition[key])#it seems already converts value list to array
                if key in self.im_keys:
                    #print('keyname: ',key)#
                    im = np.array(transition[key])#seems to be the image?
                    #print('im.shape',im.shape)#(32,2)
                    #print('im0norm',np.linalg.norm(im))
                if data is None:
                    data = np.zeros((self.size, *new_data.shape))#then fill one by one
                data[self._index] = new_data#now fill one by one#this is the data of this key!
                self.data[key] = data#the value of self.data[key] is a np array#the way to init a value in a dict

        self._index = (self._index + 1) % self.size#no more no less, just 10k pieces of data#a queue like dagger
        self._len = min(self._len + 1, self.size)#a thing saturate at self.size!
        #I think I have understood the above function!

    def sample(self, batch_size, ensemble=0):#bs=256 by default#it is sampling a few transitions!
        if ensemble == 0:#len(self) is literally self.size
            indices = np.random.randint(len(self), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(self), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extract(key, indices) for key in self.data}#106

    def samplemeancbf(self, batch_size, ensemble=0):#bs=256 by default#it is sampling a few transitions!
        if ensemble == 0:#len(self) is literally self.size
            indices = np.random.randint(len(self), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(self), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extractmean(key, indices) for key in self.data}#106

    def sample_positive(self, batch_size, key, ensemble=0):
        """
        Samples only from the entries where the array corresponding to key is nonzero
        I added this method so I could sample only from data entries in the safe set
        """
        assert len(self.data[key].shape) == 1, 'cannot sample positive from array with >1d values'
        nonzeros = self.data[key].nonzero()[0]#self.data[key] is the value#get the safe ones!
        # print(nonzeros)
        if ensemble == 0:
            indices = np.random.randint(len(nonzeros), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(nonzeros), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extract(key, nonzeros[indices]) for key in self.data}#106

    def sample_boundary_meancbf(self, batch_size, key, ensemble=0):#here my key is hvn or hvo?
        """
        Samples only from the entries where the array corresponding to key is nonzero
        I added this method so I could sample only from data entries in the safe set
        """
        assert len(self.data[key].shape) == 1, 'cannot sample positive from array with >1d values'
        #self.data[key] is a numpy array! 
        #print('key',key)#hvn
        #print('self.data[key].shape',(self.data[key]).shape)#25000
        #print('self.data[key]',self.data[key])#
        #condition=(self.data[key]<=0.0015) &(self.data[key]!=0)#4032 things!#if this then there are 4635 things!#0.002##
        #condition=(self.data[key]>=-0.0011) & (self.data[key]<=0.0015) &(self.data[key]!=0)#only 3422!#0.002#
        #condition=(self.data[key]>-0.0011) & (self.data[key]<=0.0015) &(self.data[key]!=0)#still 3422!#0.002#
        #condition=(self.data[key]>=-0.0011) & (self.data[key]<=0.0013) &(self.data[key]!=0)#only 3243!#0.002#
        #ondition=(self.data[key]>=-0.0011) & (self.data[key]<0)#only 2282!!#0.002#
        #condition=(self.data[key]>-0.001) & (self.data[key]<0)#only 2201!!#0.002#
        #condition=(self.data[key]>=-0.0011) & (self.data[key]<=0.0013) &(self.data[key]!=0)#for reacher!#only 2201!!#0.002#
        #print('condition.shape',condition.shape)
        #condition=(self.data[key]<=0.02) &(self.data[key]!=0)#this condition is for pushing!!!13.5k
        #condition=(self.data[key]<=0.025) &(self.data[key]!=0)#this condition is for pushing!!!13.889k
        #condition=(self.data[key]<=0.05) &(self.data[key]!=0)#this condition is for pushing!!!18.4k
        condition=(self.data[key]<=0.0675) &(self.data[key]!=0)#this condition is for pushing!!!18.4k#condition 1 0.0675=0.0175+0.05
        #condition=(self.data[key]<=0.0675) &(self.data[key]>0)#this condition is for pushing!!!18.4k#condition 2
        nonzeros = np.nonzero(condition)[0]#self.data[key].nonzero(condition)[0]#self.data[key] is the value#get the safe ones!
        #print('nonzeros.shape',nonzeros.shape)#2282 to 2201#(17100,) when no process!
        #print('nonzeros',nonzeros)#self.data[key]#[0 1 2 ... 17097 17098 17099]
        if ensemble == 0:
            indices = np.random.randint(len(nonzeros), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(nonzeros), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extractmean(key, nonzeros[indices]) for key in self.data}#106

    def sample_boundary_meancbf_m2(self, batch_size, key, ensemble=0):#here my key is hvn or hvo?
        """
        Samples only from the entries where the array corresponding to key is nonzero
        I added this method so I could sample only from data entries in the safe set
        """
        assert len(self.data[key].shape) == 1, 'cannot sample positive from array with >1d values'
        if key=='constraint':
            condition=(self.data[key]==1)#the colliding one!# &(self.data[key]!=0)#this condition is for pushing!!!18.4k#condition 1 0.0675=0.0175+0.05
            nonzeros = np.nonzero(condition)[0]#self.data[key].nonzero(condition)[0]#self.data[key] is the value#get the safe ones!
        else:
            nonzeros=self
        #print('nonzeros.shape',nonzeros.shape)#2282 to 2201#(17100,) when no process!
        #print('nonzeros',nonzeros)#self.data[key]#[0 1 2 ... 17097 17098 17099]
        if ensemble == 0:
            indices = np.random.randint(len(nonzeros), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(nonzeros), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extractmean(key, nonzeros[indices]) for key in self.data}#106

    def sample_boundary_m2(self, batch_size, key, ensemble=0):#here my key is hvn or hvo?
        """
        Samples only from the entries where the array corresponding to key is nonzero
        I added this method so I could sample only from data entries in the safe set
        """
        assert len(self.data[key].shape) == 1, 'cannot sample positive from array with >1d values'
        condition=(self.data[key]<=0.0675) &(self.data[key]!=0)#this condition is for pushing!!!18.4k#condition 1 0.0675=0.0175+0.05
        #condition=(self.data[key]<=0.0675) &(self.data[key]>0)#this condition is for pushing!!!18.4k#condition 2
        nonzeros = np.nonzero(condition)[0]#self.data[key].nonzero(condition)[0]#self.data[key] is the value#get the safe ones!
        #print('nonzeros.shape',nonzeros.shape)#2282 to 2201#(17100,) when no process!
        #print('nonzeros',nonzeros)#self.data[key]#[0 1 2 ... 17097 17098 17099]
        if ensemble == 0:
            indices = np.random.randint(len(nonzeros), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(nonzeros), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extract(key, nonzeros[indices]) for key in self.data}#106

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample_chunk(self, batch_size, length, ensemble=0):
        if ensemble == 0:
            idxs = np.asarray([self._sample_idx(length) for _ in range(batch_size)])
        elif ensemble > 0:
            idxs = np.asarray([[self._sample_idx(length) for _ in range(batch_size)]
                               for _ in range(ensemble)])
        else:
            raise ValueError("ensemble size cannot be negative")
        out_dict = {}
        for key in self.data:
            out = self._extract(key, idxs)#106
            out_dict[key] = out
        return out_dict

    def all_transitions(self):
        for i in range(len(self)):
            transition = {key: self.data[key] for key in self.data}
            yield transition

    def _extract(self, key, indices):#give the term you want, then return the value
        if key in self.im_keys:#obs and next_obs
            dat = self.data[key][indices]
            dat_mean, dat_log_std = np.split(dat, 2, axis=-1)
            if self.mean=='sample' or self.mean=='meancbf':#this means only the CBF is mean, other is still sampling
                dat_std = np.exp(dat_log_std)
                #print('meancbf non cbf should enter here!')
                return np.random.normal(dat_mean.squeeze(), dat_std.squeeze())#this is already sampled!
            elif self.mean=='mean':
                #print('dat_log_std',dat_log_std)#it also works for pushing! The implementation for pushing is also right!
                #print('dat_mean.shape',dat_mean.shape)
                return dat_mean.squeeze()#double check#mean latent space!
        else:#if it is not an image, then just return the value
            return self.data[key][indices]

    def _extractmean(self, key, indices):#give the term you want, then return the value
        if key in self.im_keys:#obs and next_obs
            dat = self.data[key][indices]
            dat_mean, dat_log_std = np.split(dat, 2, axis=-1)
            #print('meancbf cbf should enter here!')
            return dat_mean.squeeze()#double check
        else:#if it is not an image, then just return the value
            return self.data[key][indices]

    def _sample_idx(self, length):
        valid_idx = False
        idxs = None
        while not valid_idx:
            idx = np.random.randint(0, len(self) - length)
            idxs = np.arange(idx, idx + length) % self.size
            # Make sure data does not cross the memory index
            valid_idx = self._index not in idxs[1:]
            if 'done' in self.data:
                end = np.any(self.data['done'][idxs[:-1]])
                valid_idx = valid_idx and not end
        return idxs

    def __len__(self):
        return self._len

class EncodedReplayBuffer_expensive2:
    """
    This replay buffer uses numpy to efficiently store arbitrary data. Keys can be whatever,
    but once you push data to the buffer new data must all have the same keys (to keep parallel
    arrays parallel).

    This replay buffer replaces all images with their representation from encoder
    """
    def __init__(self, encoder, encoder2, size=10000):
        self.size = size
        self.encoder = encoder
        self.encoder2 = encoder2

        self.data = {}#finally it becomes a dict where each key's value have size number of values
        self._index = 0
        self._len = 0
        self.im_keys1 = ('obs', 'next_obs')
        self.im_keys = ('obs', 'next_obs','obs_relative','next_obs_relative')
        self.im_keys2 = ('obs_relative','next_obs_relative')

    def store_transitions(self, transitions):#transitions is 1 traj having 100 steps
        """
        Stores transitions
        :param transitions: a list of dictionaries encoding transitions. Keys can be anything
        """
        assert transitions[-1]['done'] > 0, "Last transition must be end of trajectory"
        for transition in transitions:#a transition is 1 step#It is a dictionary
            self.store_transition(transition)

    def store_transition(self, transition):#a transition is 1 step#It is a dictionary
        if len(self.data) > 0:#at first it is not like this, from second it is like this
            key_set = set(self.data)#the keys of self.data are the keys of transition!
        else:#at first it is like this
            key_set = set(transition)#you only get the keys of that dictionary! python usage!

        # assert key_set == set(transition), "Expected transition to have keys %s" % key_set

        for key in key_set:#it is a set
            data = self.data.get(key, None)#.get() is to get the value of a key

            new_data = np.array(transition[key])#it seems already converts value list to array
            if key in self.im_keys1:
                im = np.array(transition[key])#seems to be the image?
                im = ptu.torchify(im)
                new_data_mean, new_data_log_std = self.encoder(im[None] / 255)#get the latent states corresponding to the image
                new_data_mean = new_data_mean.squeeze().detach().cpu().numpy()
                new_data_log_std = new_data_log_std.squeeze().detach().cpu().numpy()
                new_data = np.dstack((new_data_mean, new_data_log_std)).squeeze()
            elif key in self.im_keys2:
                im = np.array(transition[key])#seems to be the image?
                im = ptu.torchify(im)
                new_data_mean, new_data_log_std = self.encoder2(im[None] / 255)#get the latent states corresponding to the image
                new_data_mean = new_data_mean.squeeze().detach().cpu().numpy()
                new_data_log_std = new_data_log_std.squeeze().detach().cpu().numpy()
                new_data = np.dstack((new_data_mean, new_data_log_std)).squeeze()

            if data is None:
                data = np.zeros((self.size, *new_data.shape))#then fill one by one
            data[self._index] = new_data#now fill one by one
            self.data[key] = data#the value of self.data[key] is a np array#the way to init a value in a dict

        self._index = (self._index + 1) % self.size#no more no less, just 10k pieces of data#a queue like dagger
        self._len = min(self._len + 1, self.size)#a thing saturate at self.size!
        #I think I have understood the above function!
    
    def store_dump_transition(self, transition):#a transition is 1 step#It is a dictionary
        if len(self.data) > 0:#at first it is not like this, from second it is like this
            key_set = set(self.data)#the keys of self.data are the keys of transition!
        else:#at first it is like this
            key_set = set(transition)#you only get the keys of that dictionary! python usage!

        # assert key_set == set(transition), "Expected transition to have keys %s" % key_set

        for key in key_set:#it is a set
            data = self.data.get(key, None)#.get() is to get the value of a key

            new_data = np.array(transition[key])#it seems already converts value list to array
            if key in self.im_keys1:
                im = np.array(transition[key])#seems to be the image?
                im = ptu.torchify(im)
                new_data_mean, new_data_log_std = self.encoder(im[None] / 255)#get the latent states corresponding to the image
                new_data_mean = new_data_mean.squeeze().detach().cpu().numpy()
                new_data_log_std = new_data_log_std.squeeze().detach().cpu().numpy()
                new_data = np.dstack((new_data_mean, new_data_log_std)).squeeze()
            elif key in self.im_keys2:
                im = np.array(transition[key])#seems to be the image?
                im = ptu.torchify(im)
                new_data_mean, new_data_log_std = self.encoder2(im[None] / 255)#get the latent states corresponding to the image
                new_data_mean = new_data_mean.squeeze().detach().cpu().numpy()
                new_data_log_std = new_data_log_std.squeeze().detach().cpu().numpy()
                new_data = np.dstack((new_data_mean, new_data_log_std)).squeeze()

            if data is None:
                data = np.zeros((self.size, *new_data.shape))#then fill one by one
            data[self._index] = new_data#now fill one by one
            self.data[key] = data#the value of self.data[key] is a np array#the way to init a value in a dict

        self._index = (self._index + 1) % self.size#no more no less, just 10k pieces of data#a queue like dagger
        self._len = min(self._len + 1, self.size)#a thing saturate at self.size!
        #I think I have understood the above function!
    
    def sample(self, batch_size, ensemble=0):#bs=256 by default#it is sampling a few transitions!
        if ensemble == 0:#len(self) is literally self.size
            indices = np.random.randint(len(self), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(self), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extract(key, indices) for key in self.data}#106

    def sample_positive(self, batch_size, key, ensemble=0):
        """
        Samples only from the entries where the array corresponding to key is nonzero
        I added this method so I could sample only from data entries in the safe set
        """
        assert len(self.data[key].shape) == 1, 'cannot sample positive from array with >1d values'
        nonzeros = self.data[key].nonzero()[0]#self.data[key] is the value#get the safe ones!
        # print(nonzeros)
        if ensemble == 0:
            indices = np.random.randint(len(nonzeros), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(nonzeros), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return {key: self._extract(key, nonzeros[indices]) for key in self.data}#106

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample_chunk(self, batch_size, length, ensemble=0):
        if ensemble == 0:
            idxs = np.asarray([self._sample_idx(length) for _ in range(batch_size)])
        elif ensemble > 0:
            idxs = np.asarray([[self._sample_idx(length) for _ in range(batch_size)]
                               for _ in range(ensemble)])
        else:
            raise ValueError("ensemble size cannot be negative")
        out_dict = {}
        for key in self.data:
            out = self._extract(key, idxs)#106
            out_dict[key] = out
        return out_dict

    def all_transitions(self):
        for i in range(len(self)):
            transition = {key: self.data[key] for key in self.data}
            yield transition

    def _extract(self, key, indices):#give the term you want, then return the value
        if key in self.im_keys:#obs and next_obs
            dat = self.data[key][indices]
            dat_mean, dat_log_std = np.split(dat, 2, axis=-1)
            dat_std = np.exp(dat_log_std)
            return np.random.normal(dat_mean.squeeze(), dat_std.squeeze())
        else:#if it is not an image, then just return the value
            return self.data[key][indices]

    def _sample_idx(self, length):
        valid_idx = False
        idxs = None
        while not valid_idx:
            idx = np.random.randint(0, len(self) - length)
            idxs = np.arange(idx, idx + length) % self.size
            # Make sure data does not cross the memory index
            valid_idx = self._index not in idxs[1:]
            if 'done' in self.data:
                end = np.any(self.data['done'][idxs[:-1]])
                valid_idx = valid_idx and not end
        return idxs

    def __len__(self):
        return self._len
