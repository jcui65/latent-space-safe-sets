import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.model import GenericNet, GenericNetcbf
from .interfaces import EncodedModule

import torch
import torch.nn as nn
from torch.autograd.functional import hessian
from torch.nn.utils import _stateless

class CBFdotEstimatorlatentplana(nn.Module, EncodedModule):#supervised learning very similar to gi or constraint estimator
    """
    Simple constraint predictor using binary cross entropy
    """

    def __init__(self, encoder, params: dict):
        """
        Initializes a constraint estimator
        """
        super(CBFdotEstimatorlatentplana, self).__init__()
        EncodedModule.__init__(self, encoder)

        self.d_obs = params['d_obs']#(3,64,64)#dimension of observation
        self.d_latent = params['d_latent']#32#4#2+2#
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
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)#,weight_decay=0.0001)#0.001)#0.1)#0.01)#1.0)#

    def forward(self, obs, already_embedded=False):
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
        #action=ptu.torchify(action)
        #print('embedding',embedding)
        #print('embedding.shape', embedding.shape)#torch.Size([256, 32])
        #print('action',action)
        #print('action.shape', action.shape)# torch.Size([256, 2])
        #ea=torch.concat((embedding,action),-1)
        log_probs = self.net(embedding)#self.net(ea)#self.net(embedding)#why 3 kinds of sizes?
        return log_probs

    def cbfdots(self, obs, already_embedded=False):#the forward function for numpy input#this is used in plotting
        obs = ptu.torchify(obs)
        #embedding = self.encoder.encode(obs).detach()  # workaround#currently I am in the state space
        #print('embedding.shape',embedding.shape)# torch.Size([180, 32])
        #device = embedding.device
        ##device = obs.device
        ##zero2=torch.zeros((embedding.shape[0],2)).to(device)
        ##zero2 = torch.zeros((obs.shape[0], 2)).to(device)
        ##action=zero2
        ##ea0=torch.concat((embedding,zero2),1)
        #logits = self(ea0,action, already_embedded=True)
        logits = self(obs,already_embedded)#self(obs, action)#
        probs = logits#torch.sigmoid(logits)#
        return ptu.to_numpy(probs)

    def cbfdots_planb(self, obs, already_embedded=False):#the forward function for numpy input#this is used in plotting
        obs = ptu.torchify(obs)
        #embedding = self.encoder.encode(obs).detach()  # workaround#currently I am in the state space
        #print('embedding.shape',embedding.shape)# torch.Size([180, 32])
        #device = embedding.device
        ##device = obs.device
        ##zero2=torch.zeros((embedding.shape[0],2)).to(device)
        ##zero2 = torch.zeros((obs.shape[0], 2)).to(device)
        ##action=zero2
        ##ea0=torch.concat((embedding,zero2),1)
        #logits = self(ea0,action, already_embedded=True)
        #logits = self(obs,already_embedded)#self(obs, action)#
        #probs = logits#torch.sigmoid(logits)#
        if not already_embedded:
            embedding = self.encoder.encode(obs).detach()#workaround#currently I am in the state space
        else:
            embedding = obs
        #print('embedding', embedding)
        #print('embedding.shape', embedding.shape)#torch.Size([180, 32])
        cbfvalues=embedding[:,-1]
        #print('cbfvalues', cbfvalues)
        #print('cbfvalues.shape',cbfvalues.shape)#torch.Size([32])
        return ptu.to_numpy(cbfvalues)

    def update(self, next_obs, constr, already_embedded=False):#the training process
        self.trained = True
        next_obs = ptu.torchify(next_obs)#input
        constr = ptu.torchify(constr)#output

        self.optimizer.zero_grad()
        loss = self.loss(next_obs, constr, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), {'cbfd': loss.item()}

    def loss(self, next_obs, constr, already_embedded=False):
        #next_obs=torch.autograd.Variable(next_obs,requires_grad=True)
        logits = self(next_obs, already_embedded).squeeze()#.forward!#prediction
        #print('logits',logits)
        targets = constr#label
        loss1 = self.loss_func(logits, targets)
        #external_grad = torch.ones_like(loss1)
        #loss1.backward(gradient=external_grad)
        #logits.mean().backward(retain_graph=True)#(torch.ones_like(loss1))#
        #gv=next_obs.grad
        #print('gv',gv)
        #epsilon=250#200#
        #print('torch.norm(gv).item()',torch.norm(gv).item())
        #model = self.net #CBFdotEstimatorlatentplana(encoder, params).to(ptu.TORCH_DEVICE)#model = torch.nn.Linear(2, 2)#
        #inp = torch.rand(1, 2)

        #names = list(n for n, _ in model.named_parameters())#list(n for n, _ in model.named_parameters())
        #print('names',names)
        #hv = hessian(model,next_obs)
        #hv=hessian(lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, next_obs),next_obs)
                 #tuple(model.parameters()))
        #loss2=0.0000001*torch.clamp(torch.norm(gv)-epsilon,0)#0.01*max((torch.norm(gv).item()-epsilon),0)
        #print('loss2',loss2)
        #if (torch.norm(gv).item()-epsilon)<=0:
            #loss2=0
        #else:
            #loss2=0.01*(torch.norm(gv).item()-epsilon)
        return loss1#+loss2#

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
