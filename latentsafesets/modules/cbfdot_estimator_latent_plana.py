import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.model import GenericNet, GenericNetcbf, GenericNetcbfelu
from .interfaces import EncodedModule

import torch
import torch.nn as nn
from torch.autograd.functional import jacobian,hessian
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

        self.d_obs = params['d_obs']#(3,3,64,64) in reacher#(3,64,64) in spb and pushing#dimension of observation#
        self.d_latent = params['d_latent']#32#4#2+2#
        self.batch_size = params['cbfd_batch_size']#256
        self.targ_update_counter = 0
        self.loss_func = torch.nn.SmoothL1Loss()#a regression loss#designate the loss function#torch.nn.BCEWithLogitsLoss()#
        self.trained = False
        #self.net = GenericNet(self.d_latent, 1, params['cbfd_n_hidden'], params['cbfd_hidden_size']).to(ptu.TORCH_DEVICE)#the network that uses relu activation
        #self.net = GenericNetcbf(self.d_obs, 1, params['cbfd_n_hidden'],params['cbfd_hidden_size']).to(ptu.TORCH_DEVICE)
        #self.net = GenericNetcbf(self.d_latent, 1, params['cbfd_n_hidden'], params['cbfd_hidden_size']).to(
            #ptu.TORCH_DEVICE)
        self.net = GenericNetcbfelu(self.d_latent, 1, params['cbfd_n_hidden'], params['cbfd_hidden_size']).to(
            ptu.TORCH_DEVICE)
        #print(self.net)#input size 4, output size 1#the network that uses the tanh activation
        lr = params['cbfd_lr']
        if params['train_cbf']=='no2':
            lr=0
        print('learning rate: %f'%(lr))#it is 0 if 'no2'!!!
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)#,weight_decay=0.0001)#0.001)#0.1)#0.01)#1.0)#
        self.mean=params['mean']
    def forward(self, obs, already_embedded=False):
        """
        Returns inputs to sigmoid for probabilities
        """
        #print('obs.shape',obs.shape)
        if not already_embedded:
            #embedding = self.encoder.encode(obs).detach()#workaround#currently I am in the state space
            if self.mean=='sample':
                embedding = self.encoder.encode(obs).detach()
                #print('get sample embedding from images! embedding.shape',embedding.shape)
            elif self.mean=='mean':
                embedding = self.encoder.encodemean(obs).detach()
                #print('get mean embedding from images! embedding.shape',embedding.shape)#
        else:
            embedding = obs
            #print('get embedded embedding already! embedding.shape',embedding.shape)#torch.Size([20, 500, 1, 32]), torch.Size([1, 32]), torch.Size([20, 500, 3, 32])
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
            #embedding = self.encoder.encode(obs).detach()#workaround#currently I am in the state space
            if self.mean=='sample':
                embedding = self.encoder.encode(obs).detach()
            elif self.mean=='mean':
                embedding = self.encoder.encodemean(obs).detach()
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
        #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
        constr = ptu.torchify(constr)#output

        self.optimizer.zero_grad()
        loss = self.loss(next_obs, constr, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), {'cbfd': loss.item()}#the first term is already the item

    def loss(self, next_obs, constr, already_embedded=False):
        #next_obs=torch.autograd.Variable(next_obs,requires_grad=True)
        logits = self(next_obs, already_embedded).squeeze()#.forward!#prediction
        #print('logits',logits)#the value of the CBF
        targets = constr#label
        loss1 =1000000*self.loss_func(logits, targets)#+jacobian(self.forward)-#1000000 for reacher
        #print('next_obs.shape',next_obs.shape)
        #selfforwardtrue=lambda nextobs: self(nextobs, True)
        #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
        #jno=jacobian(selfforwardtrue,next_obs,create_graph=True)#jno means jacobian next_obs
        #print('jno.shape',jno.shape)#torch.Size([256, 1, 256, 32])
        '''
        hno=torch.zeros((next_obs.shape[0],next_obs.shape[1],next_obs.shape[1]))#hno means hessian next observation
        for i in range(next_obs.shape[0]):
            hnoi=hessian(selfforwardtrue, next_obs[i], create_graph=True) #it is zero!
            #print('hnoi',hnoi)
            #print('hnoi.shape',hnoi.shape)#(32,32)
            hno[i]=hnoi
            '''
        #jno=hessian(selfforwardtrue, next_obs, create_graph=True)  # jno means jacobian next_obs
        #print('jno',jno)
        #jnon=torch.norm(jno)#jnon means  norm of jacobian next_obs
        #jnon=torch.norm(hno)
        #print('jnon',jnon)
        #epsilon=1e-6#1e-5#1e-4#1#this is for showing the effectiveness of the loss term on the magnitude of the gradient#1e-3#1e-2#this is for showing the effectiveness of the loss term on the magnitude of the gradient#1e-6#
        #loss2=epsilon*jnon
        #print('loss2',loss2.item())#the main point is on the global coordinate's case
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
