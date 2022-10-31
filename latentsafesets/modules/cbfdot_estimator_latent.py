import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.model import GenericNet, GenericNetcbf
from .interfaces import EncodedModule

import torch
import torch.nn as nn


class CBFdotEstimatorlatent(nn.Module, EncodedModule):#supervised learning very similar to gi or constraint estimator
    """
    Simple constraint predictor using binary cross entropy
    """

    def __init__(self, encoder, params: dict):
        """
        Initializes a constraint estimator
        """
        super(CBFdotEstimatorlatent, self).__init__()
        EncodedModule.__init__(self, encoder)

        self.d_obs = params['d_obs']#(3,64,64)#dimension of observation
        self.d_latent = 2+params['d_latent']#32#4#2+2#
        self.batch_size = params['cbfd_batch_size']#256
        self.targ_update_counter = 0
        self.loss_func = torch.nn.SmoothL1Loss()#a regression loss#designate the loss function#torch.nn.BCEWithLogitsLoss()#
        self.trained = False
        #self.net = GenericNet(self.d_latent, 1, params['cbfd_n_hidden'], params['cbfd_hidden_size']).to(ptu.TORCH_DEVICE)#the network that uses relu activation
        #self.net = GenericNetcbf(self.d_obs, 1, params['cbfd_n_hidden'],params['cbfd_hidden_size']).to(ptu.TORCH_DEVICE)
        self.net = GenericNetcbf(self.d_latent, 1, params['cbfd_n_hidden'], params['cbfd_hidden_size']).to(
            ptu.TORCH_DEVICE)
        #print(self.net)#input size 4, output size 1#the network that uses the tanh activation
        lr = params['cbfd_lr']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, obs, action, already_embedded=False):
        """
        Returns inputs to sigmoid for probabilities
        """
        #print('obs.shape',obs.shape)
        if not already_embedded:
            embedding = self.encoder.encode(obs).detach()#workaround#currently I am in the state space
        else:
            embedding = obs
        #print('embedding.shape',embedding.shape)#torch.Size([1000,5,4])#torch.Size([256,4])#torch.Size([180,4])#
        #device = embedding.device
        action=ptu.torchify(action)
        #print('embedding',embedding)
        #print('embedding.shape', embedding.shape)#torch.Size([256, 32])
        #print('action',action)
        #print('action.shape', action.shape)# torch.Size([256, 2])
        ea=torch.concat((embedding,action),-1)
        log_probs = self.net(ea)#self.net(embedding)#why 3 kinds of sizes?
        return log_probs

    def cbfdots(self, obs,already_embedded=False):#the forward function for numpy input#this is used in plotting
        obs = ptu.torchify(obs)
        #embedding = self.encoder.encode(obs).detach()  # workaround#currently I am in the state space
        #print('embedding.shape',embedding.shape)# torch.Size([180, 32])
        #device = embedding.device
        device = obs.device
        #zero2=torch.zeros((embedding.shape[0],2)).to(device)
        zero2 = torch.zeros((obs.shape[0], 2)).to(device)
        action=zero2
        #ea0=torch.concat((embedding,zero2),1)
        #logits = self(ea0,action, already_embedded=True)
        logits = self(obs, action,already_embedded)
        probs = logits#torch.sigmoid(logits)#
        return ptu.to_numpy(probs)

    def update(self, next_obs,action, constr, already_embedded=False):#the training process
        self.trained = True
        next_obs = ptu.torchify(next_obs)#input
        constr = ptu.torchify(constr)#output

        self.optimizer.zero_grad()
        loss = self.loss(next_obs, action,constr, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), {'cbfd': loss.item()}

    def loss(self, next_obs,action, constr, already_embedded=False):
        logits = self(next_obs, action,already_embedded).squeeze()#.forward!#prediction
        targets = constr#label
        loss = self.loss_func(logits, targets)
        return loss

    def step(self):
        """
        This assumes you've already done backprop. Steps optimizers
        """
        self.optimizer.step()

    def save(self, file):
        torch.save(self.net.state_dict(), file)

    def load(self, file):
        from latentsafesets.utils.pytorch_utils import TORCH_DEVICE
        self.net.load_state_dict(torch.load(file, map_location=TORCH_DEVICE))
        self.trained = True
