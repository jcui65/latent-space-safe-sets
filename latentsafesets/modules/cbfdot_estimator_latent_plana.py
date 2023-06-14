import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.model import GenericNet, GenericNetcbf, GenericNetcbfelu
from .interfaces import EncodedModule

import torch
import torch.nn as nn
from torch.autograd.functional import jacobian,hessian
from torch.nn.utils import _stateless
import logging
log = logging.getLogger("cbfd train")

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
        self.reg_lipschitz=params['reg_lipschitz']
        self.env=params['env']
        self.gammasafe=params['gammasafe']
        self.gammaunsafe=params['gammaunsafe']
        self.gammadyn=params['gammadyn']
        self.alpha=params['cbfdot_thresh']
        self.dz=1*params['sigmaz']*torch.ones((self.d_latent),device=ptu.TORCH_DEVICE)#it needs to be a tensor!#just first use 1 sigma_z to make it feasible!
        self.noofsigmadhz=params['noofsigmadhz']
        self.dhz=params['dhz']#you set it to be 0.00002, it is OK I think
        if self.env=='reacher':
            self.lipthres=1/200#tune this parameter!
        elif self.env=='push':
            self.lipthres=1/100#1/500#will subject to change!
        elif self.env=='spb':
            self.lipthres=15#5#
        self.w1=params['w1']
        self.w2=params['w2']
        self.w3=params['w3']
        self.w4=params['w4']
        self.w5=params['w5']#10#50#
        self.w6=params['w6']
        self.w7=params['w7']
        self.w8=params['w8']
        self.w11=params['w11']#w1 for online
        self.w12=params['w12']#w2 for online
        self.w13=params['w13']#w3 for online
        self.w14=params['w14']#w4 for online
        self.w15=params['w15']#w4 for online
        self.stepstohell=params['stepstohell']
        self.m10=1e-8#for better visualization#1e-10#
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
            elif self.mean=='mean' or self.mean=='meancbf':
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
            elif self.mean=='mean' or self.mean=='meancbf':
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
        loss,data = self.loss(next_obs, constr, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), data##loss.item(), {'cbfd': loss.item()}#the first term is already the item

    def update_m2s(self, obs, next_obs, constr, already_embedded=False):#the training process
        self.trained = True
        obs = ptu.torchify(obs)#input
        next_obs = ptu.torchify(next_obs)#input
        #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
        constr = ptu.torchify(constr)#output

        self.optimizer.zero_grad()
        #loss = self.loss(next_obs, constr, already_embedded)
        loss,data = self.lossm2s(obs,next_obs, constr, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), data##loss.item(), {'cbfd': loss.item()}#the first term is already the item

    def update_m2u(self, obs, next_obs, constr, already_embedded=False):#the training process
        self.trained = True
        obs = ptu.torchify(obs)#input
        next_obs = ptu.torchify(next_obs)#input
        #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
        constr = ptu.torchify(constr)#output

        self.optimizer.zero_grad()
        #loss = self.loss(next_obs, constr, already_embedded)
        loss,data = self.lossm2u(obs,next_obs, constr, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), data##loss.item(), {'cbfd': loss.item()}#the first term is already the item

    def update_m2s_0109(self, obs, next_obs, constr, already_embedded=False):#the training process
        self.trained = True
        obs = ptu.torchify(obs)#input
        next_obs = ptu.torchify(next_obs)#input
        #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
        constr = ptu.torchify(constr)#output
        cbfv=constr*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
        self.optimizer.zero_grad()
        #loss = self.loss(next_obs, constr, already_embedded)
        #loss,data = self.lossm2s(obs,next_obs, constr, already_embedded)
        loss,data = self.lossm2s0109(obs,next_obs, cbfv, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), data##loss.item(), {'cbfd': loss.item()}#the first term is already the item

    def update_m2s_0109_online(self, obs, next_obs, constr, already_embedded=False):#the training process
        self.trained = True
        obs = ptu.torchify(obs)#input
        next_obs = ptu.torchify(next_obs)#input
        #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
        constr = ptu.torchify(constr)#output
        cbfv=constr*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
        self.optimizer.zero_grad()
        #loss = self.loss(next_obs, constr, already_embedded)
        #loss,data = self.lossm2s(obs,next_obs, constr, already_embedded)
        loss,data = self.lossm2s0109online(obs,next_obs, cbfv, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), data##loss.item(), {'cbfd': loss.item()}#the first term is already the item

    def update_m2u_0109(self, obs, next_obs, constr, already_embedded=False):#the training process
        self.trained = True
        obs = ptu.torchify(obs)#input
        next_obs = ptu.torchify(next_obs)#input
        #print('next_obs.shape',next_obs.shape)#448,32#torch.Size([256, 32])
        constr = ptu.torchify(constr)#output
        cbfv=constr*(-self.gammasafe-self.gammaunsafe)+self.gammasafe
        self.optimizer.zero_grad()
        #loss = self.loss(next_obs, constr, already_embedded)
        #loss,data = self.lossm2u(obs,next_obs, constr, already_embedded)
        loss,data = self.lossm2u0109(obs,next_obs, cbfv, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), data##loss.item(), {'cbfd': loss.item()}#the first term is already the item

    def loss(self, next_obs, constr, already_embedded=False):
        #if constr==0:
        #next_obs=torch.autograd.Variable(next_obs,requires_grad=True)
        logits = self(next_obs, already_embedded).squeeze()#.forward!#prediction
        #print('logits',logits)#the value of the CBF
        targets = constr#label
        loss1 =1000000*self.loss_func(logits, targets)#+jacobian(self.forward)-#1000000 for reacher
        #print('next_obs.shape',next_obs.shape)
        if self.reg_lipschitz=='yes':
            selfforwardtrue=lambda nextobs: self(nextobs, True)
            #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
            jno=jacobian(selfforwardtrue,next_obs,create_graph=True)#jno means jacobian next_obs
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
            jnon=torch.norm(jno)#jnon means  norm of jacobian next_obs
            #jnon=torch.norm(hno)
            #print('jnon',jnon)
            #epsilon=1e-6#1e-5#1e-4#1#this is for showing the effectiveness of the loss term on the magnitude of the gradient#1e-3#1e-2#this is for showing the effectiveness of the loss term on the magnitude of the gradient#1e-6#
            lamb=50#100#10#I set this to be 50/100!
            if self.env=='reacher':
                lipthres=1/900
            elif self.env=='push':
                lipthres=1/500
            loss2=lamb*torch.nn.functional.relu(jnon-lipthres)#I set it to be 1/900
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
        else:
            device=loss1.device#just as a place holder!
            loss2=torch.zeros(1,device=device)
        loss=loss1+loss2##
        data = {
            'cbf_total': loss.item(),
            'cbf': loss1.item(),
            'regularization': loss2.item()}

        return loss,data

    def lossm2s(self,obs, next_obs, cbfv, already_embedded=False):
        cbfold=self(obs, already_embedded).squeeze()#.forward!#prediction
        cbfnew = self(next_obs, already_embedded).squeeze()#size 128#.forward!#prediction
        #print('logits',logits)#the value of the CBF
        targets = cbfv#constr#some function of constr#label
        loss1 =torch.nn.functional.relu(targets-cbfold)#cbfold should be greater than target value! verified!
        loss1=torch.mean(loss1)#1000000*self.loss_func(logits, targets)#+jacobian(self.forward)-#1000000 for reacher
        #print('loss1.shape',loss1.shape)#used to be 128
        loss2 =torch.nn.functional.relu(targets-cbfnew)#cbfnew should be greater than the target value! verified!
        loss2=torch.mean(loss2)##
        #print('loss2.shape',loss2.shape)#used to be 128
        normdiffthres=(self.gammasafe+self.gammaunsafe)/self.stepstohell#15#I PICK IT TO BE 15#0.05#?
        loss4=torch.nn.functional.relu(torch.abs(cbfnew-cbfold)-normdiffthres)#the difference should be less than that!!!
        loss4=torch.mean(loss4)#
        #print('loss4.shape',loss4.shape)#used to be 128
        #print('next_obs.shape',next_obs.shape)#(128,32)
        if self.reg_lipschitz=='yes':
            selfforwardtrue=lambda nextobs: self(nextobs, True)
            #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
            jno=torch.zeros_like(next_obs)
            for i in range(next_obs.shape[0]):
                jnoi=jacobian(selfforwardtrue,next_obs[i],create_graph=True)#jnoi means jacobian next_obs ith
                jno[i]=jnoi#jnoi should be 32 dimensional
            #jnon=torch.norm(jno)#jnon means  norm of jacobian next_obs
            #jnon=torch.norm(jno,dim=-1)#128 now!
            #print('jno.shape',jno.shape)#jno.shape torch.Size([128, 1, 128, 32])
            #print('jnon.shape',jnon.shape)#it use to be a scalar, 0.6009, with shape 0
            #loss5=torch.nn.functional.relu(jnon-self.lipthres)
            loss5=0*loss4##torch.mean(loss5)#I set it to be 1/900
            bztut=cbfnew+(self.alpha-1)*cbfold-torch.matmul(jno,self.dz)#.dot(jno,self.dz)#torch.dot(jno,self.dz) should be a scalar#jno*self.dz
            qztut=bztut-(2-self.alpha)*self.dhz#cbfnew has its first dimension to be 128
            loss3=torch.nn.functional.relu(self.gammadyn-qztut)#this means that qztut should be smaller than gammadyn
            loss3=torch.mean(loss3)#0#make it a CBF#finally!
            #print('loss3.shape',loss3.shape)#used to be 128
        else:
            loss5=0*loss4#make it a zero tensor!
            loss3=0*loss4
        #print('loss5.shape',loss5.shape)#used to be 128
        #print('cbfnew.shape',cbfnew.shape)#shape 128
        #print('self.dz.shape',self.dz.shape)#shape 32
        
        loss=self.w1*loss1+self.w2*loss2+self.w3*loss3+self.w4*loss4+self.w5*loss5##
        data = {
            'cbf_total': loss.item(),
            'old_safe': max(self.w1*loss1.item(),self.m10),#old safe
            'new_safe': max(self.w2*loss2.item(),self.m10),#for the granularity of plotting
            'old_unsafe':self.m10,#want to show the log plots!#0,#
            'new_unsafe':self.m10,#0,#
            'make_it_a_cbf':max(self.w3*loss3.item(),self.m10),
            'closeness_safe':max(self.w4*loss4.item(),self.m10),
            'closeness_unsafe':self.m10,
            'regularization':self.w5*loss5.item()}
        return loss,data

    def lossm2u(self,obs, next_obs, cbfv, already_embedded=False):
        cbfold=self(obs, already_embedded).squeeze()#.forward!#prediction
        cbfnew = self(next_obs, already_embedded).squeeze()#.forward!#prediction
        #print('logits',logits)#the value of the CBF
        targets = cbfv#constr#some function of constr#label#if target<0 (violation), then cbfold should be less than it. Otherwise,
        loss1=torch.where(targets<0,torch.nn.functional.relu(cbfold-targets),0*torch.nn.functional.relu(cbfold-targets))#128 dim
        #print('loss1',loss1)#cbf should be less than the target!
        loss2=torch.where(targets<0,torch.nn.functional.relu(cbfnew-targets),0*torch.nn.functional.relu(cbfnew-targets))#128 dim
        count=torch.count_nonzero(targets<0)#I don't do a zero handling, as I think it is very unlikely to have such case
        #log.info('count:%d'%(count))#it should be something between 1 and 128
        if count==0:
            count+=1#just in case!
        loss1=torch.sum(loss1)/count#torch.mean(loss1)#this mean operation is a diluting factor!!!
        #log.info('loss1scalar:%f'%(loss1.item()))#sanity check passed!
        loss2=torch.sum(loss2)/count#torch.mean(loss2)#I should mean over all the negative label samples!
        #if targets<0:#you meet the unsafe point!
            #loss1 =torch.nn.functional.relu(cbfold-targets)#1000000*self.loss_func(logits, targets)#+jacobian(self.forward)-#1000000 for reacher
            #loss2 =torch.nn.functional.relu(cbfnew-targets)#
        #elif targets>=0:
            #loss1=0*torch.nn.functional.relu(targets-cbfold)#don't update this!
            #loss2=0*torch.nn.functional.relu(targets-cbfnew)#just for consistency!
        normdiffthres=(self.gammasafe+self.gammaunsafe)/self.stepstohell#15#I PICK IT TO BE 15#0.05#?
        #loss4=torch.nn.functional.relu(torch.abs(cbfnew-cbfold)-normdiffthres)#
        loss41=torch.nn.functional.relu(cbfnew-cbfold)#I want cbfnew<cbfold in this case
        loss42=torch.nn.functional.relu(cbfold-cbfnew-normdiffthres)#I want cbfnew<cbfold not too much!
        loss43=torch.nn.functional.relu(torch.abs(cbfnew-cbfold)-normdiffthres)#I want the difference of the 2 not too much!
        loss4=torch.where(targets<0,loss41+loss42,loss43)#if targets<0, then this means that you have a violation and then you should use loss41+42
        loss4=torch.mean(loss4)#if targets>0, then you not yet encounter a violation, then you just need to use loss43
        #print('next_obs.shape',next_obs.shape)
        if self.reg_lipschitz=='yes':
            #selfforwardtrue=lambda nextobs: self(nextobs, True)
            #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
            '''
            jno=jacobian(selfforwardtrue,next_obs,create_graph=True)#jno means jacobian next_obs
            jnon=torch.norm(jno)#jnon means  norm of jacobian next_obs
            if self.env=='reacher':
                lipthres=1/900
            elif self.env=='push':
                lipthres=1/500
            elif self.env=='spb':
                lipthres=1/5
            loss5=torch.nn.functional.relu(jnon-lipthres)#I set it to be 1/900
            '''
            '''
            jno=torch.zeros_like(next_obs)
            for i in range(next_obs.shape[0]):
                jnoi=jacobian(selfforwardtrue,next_obs[i],create_graph=True)#jnoi means jacobian next_obs ith
                jno[i]=jnoi#jnoi should be 32 dimensional
            #jnon=torch.norm(jno)#jnon means  norm of jacobian next_obs
            jnon=torch.norm(jno,dim=-1)#128 now!
            #print('jno.shape',jno.shape)#jno.shape torch.Size([128, 1, 128, 32])
            #print('jnon.shape',jnon.shape)#it use to be a scalar, 0.6009, with shape 0
            loss5=torch.nn.functional.relu(jnon-self.lipthres)
            '''
            loss5=0*loss4#torch.mean(loss5)#
        else:
            loss5=0*loss4
        loss=self.w6*loss1+self.w7*loss2+self.w8*loss4+self.w5*loss5##
        data = {
            'cbf_total': loss.item(),
            'old_safe':self.m10,#want to show the log plots!#0,#
            'new_safe':self.m10,#0,#
            'old_unsafe': max(self.w6*loss1.item(),self.m10),#old safe
            'new_unsafe': max(self.w7*loss2.item(),self.m10),
            'make_it_a_cbf':self.m10,#want to show the log plots!#0,#0,#-0.001,#just for consistency in plotting!
            'closeness_safe':self.m10,
            'closeness_unsafe':max(self.w8*loss4.item(),self.m10),#for plotting!#
            'regularization':self.w5*loss5.item()}
        return loss,data

    def lossm2s0109(self,obs, next_obs, cbfv, already_embedded=False):
        cbfold=self(obs, already_embedded).squeeze()#.forward!#prediction
        cbfnew = self(next_obs, already_embedded).squeeze()#size 128#.forward!#prediction
        #print('logits',logits)#the value of the CBF
        targets = cbfv#constr#some function of constr#label
        loss1 =torch.nn.functional.relu(targets-cbfold)#cbfold should be greater than target value! verified!
        loss1=torch.mean(loss1)#1000000*self.loss_func(logits, targets)#+jacobian(self.forward)-#1000000 for reacher
        #print('loss1.shape',loss1.shape)#used to be 128
        loss2 =torch.nn.functional.relu(targets-cbfnew)#cbfnew should be greater than the target value! verified!
        loss2=torch.mean(loss2)##
        #print('loss2.shape',loss2.shape)#used to be 128
        normdiffthres=(self.gammasafe+self.gammaunsafe)/self.stepstohell#15#I PICK IT TO BE 15#0.05#?
        loss4=torch.nn.functional.relu(torch.abs(cbfnew-cbfold)-normdiffthres)#the difference should be less than that!!!
        loss4=torch.mean(loss4)#
        #print('loss4.shape',loss4.shape)#used to be 128
        #print('next_obs.shape',next_obs.shape)#(128,32)
        if self.reg_lipschitz=='yes':
            #selfforwardtrue=lambda nextobs: self(nextobs, True)
            #print('next_obs.shape',next_obs.shape)#torch.Size([256, 32])
            #jno=torch.zeros_like(next_obs)
            #for i in range(next_obs.shape[0]):
                #jnoi=jacobian(selfforwardtrue,next_obs[i],create_graph=True)#jnoi means jacobian next_obs ith
                #jno[i]=jnoi#jnoi should be 32 dimensional
            #jnon=torch.norm(jno)#jnon means  norm of jacobian next_obs
            #jnon=torch.norm(jno,dim=-1)#128 now!
            #print('jno.shape',jno.shape)#jno.shape torch.Size([128, 1, 128, 32])
            #print('jnon.shape',jnon.shape)#it use to be a scalar, 0.6009, with shape 0
            #loss5=torch.nn.functional.relu(jnon-self.lipthres)
            loss5=0*loss4##torch.mean(loss5)#I set it to be 1/900
            bztut=cbfnew+(self.alpha-1)*cbfold#-torch.matmul(jnon,self.dz)#.dot(jno,self.dz)#torch.dot(jno,self.dz) should be a scalar#jno*self.dz
            qztut=bztut-(2-self.alpha)*self.dhz*self.noofsigmadhz#it is finally right now!#cbfnew has its first dimension to be 128
            loss3=torch.nn.functional.relu(self.gammadyn-qztut)#this means that qztut should be smaller than gammadyn
            loss3=torch.mean(loss3)#0#make it a CBF#finally!
            #print('loss3.shape',loss3.shape)#used to be 128
        else:
            loss5=0*loss4#make it a zero tensor!
            loss3=0*loss4
        #print('loss5.shape',loss5.shape)#used to be 128
        #print('cbfnew.shape',cbfnew.shape)#shape 128
        #print('self.dz.shape',self.dz.shape)#shape 32
        
        loss=self.w1*loss1+self.w2*loss2+self.w3*loss3+self.w4*loss4+self.w5*loss5##
        data = {
            'cbf_total': loss.item(),
            'old_safe': max(self.w1*loss1.item(),self.m10),#old safe
            'new_safe': max(self.w2*loss2.item(),self.m10),#for the granularity of plotting
            'old_unsafe':self.m10,#want to show the log plots!#0,#
            'new_unsafe':self.m10,#0,#
            'make_it_a_cbf':max(self.w3*loss3.item(),self.m10),
            'closeness_safe':max(self.w4*loss4.item(),self.m10),
            'closeness_unsafe':self.m10,
            'regularization':self.w5*loss5.item()}
        return loss,data

    def lossm2s0109online(self,obs, next_obs, cbfv, already_embedded=False):
        cbfold=self(obs, already_embedded).squeeze()#.forward!#prediction
        cbfnew = self(next_obs, already_embedded).squeeze()#size 128#.forward!#prediction
        #print('logits',logits)#the value of the CBF
        targets = cbfv#constr#some function of constr#label
        loss1 =torch.nn.functional.relu(targets-cbfold)#cbfold should be greater than target value! verified!
        loss1=torch.mean(loss1)#1000000*self.loss_func(logits, targets)#+jacobian(self.forward)-#1000000 for reacher
        #print('loss1.shape',loss1.shape)#used to be 128
        loss2 =torch.nn.functional.relu(targets-cbfnew)#cbfnew should be greater than the target value! verified!
        loss2=torch.mean(loss2)##
        #print('loss2.shape',loss2.shape)#used to be 128
        normdiffthres=(self.gammasafe+self.gammaunsafe)/self.stepstohell#15#I PICK IT TO BE 15#0.05#?
        loss4=torch.nn.functional.relu(torch.abs(cbfnew-cbfold)-normdiffthres)#the difference should be less than that!!!
        loss4=torch.mean(loss4)#
        #print('loss4.shape',loss4.shape)#used to be 128
        #print('next_obs.shape',next_obs.shape)#(128,32)
        if self.reg_lipschitz=='yes':
            loss5=0*loss4##torch.mean(loss5)#I set it to be 1/900
            bztut=cbfnew+(self.alpha-1)*cbfold#-torch.matmul(jnon,self.dz)#.dot(jno,self.dz)#torch.dot(jno,self.dz) should be a scalar#jno*self.dz
            qztut=bztut-(2-self.alpha)*self.dhz*self.noofsigmadhz#it is finally right now!#cbfnew has its first dimension to be 128
            loss3=torch.nn.functional.relu(self.gammadyn-qztut)#this means that qztut should be smaller than gammadyn
            loss3=torch.mean(loss3)#0#make it a CBF#finally!
            #print('loss3.shape',loss3.shape)#used to be 128
        else:
            loss5=0*loss4#make it a zero tensor!
            loss3=0*loss4
        #print('loss5.shape',loss5.shape)#used to be 128
        #print('cbfnew.shape',cbfnew.shape)#shape 128
        #print('self.dz.shape',self.dz.shape)#shape 32
        loss=self.w11*loss1+self.w12*loss2+self.w13*loss3+self.w14*loss4+self.w15*loss5##
        data = {
            'cbf_total': loss.item(),
            'old_safe': max(self.w11*loss1.item(),self.m10),#old safe
            'new_safe': max(self.w12*loss2.item(),self.m10),#for the granularity of plotting
            'old_unsafe':self.m10,#want to show the log plots!#0,#
            'new_unsafe':self.m10,#0,#
            'make_it_a_cbf':max(self.w13*loss3.item(),self.m10),
            'closeness_safe':max(self.w14*loss4.item(),self.m10),
            'closeness_unsafe':self.m10,
            'regularization':self.w15*loss5.item()}
        return loss,data

    def lossm2u0109(self,obs, next_obs, cbfv, already_embedded=False):
        cbfold=self(obs, already_embedded).squeeze()#.forward!#prediction
        cbfnew = self(next_obs, already_embedded).squeeze()#.forward!#prediction
        #print('logits',logits)#the value of the CBF#targets<self.gammasafe is important! You cannot assign a equal sign to it! You even need to add margin!
        targets = cbfv#constr#some function of constr#label#if target<0 (violation), then cbfold should be less than it. Otherwise,
        loss1=torch.where(targets<self.gammasafe,torch.nn.functional.relu(cbfold-targets),0*torch.nn.functional.relu(cbfold-targets))#128 dim
        #print('loss1',loss1)#cbf should be less than the target!
        loss2=torch.where(targets<self.gammasafe,torch.nn.functional.relu(cbfnew-targets),0*torch.nn.functional.relu(cbfnew-targets))#128 dim
        count=torch.count_nonzero(targets<self.gammasafe)#0)#I don't do a zero handling, as I think it is very unlikely to have such case
        #log.info('count:%d'%(count))#it should be something between 1 and 128
        if count==0:#sample positive will be useful here!!!
            count+=1#just in case!#Now it is a must!
        loss1=torch.sum(loss1)/count#torch.mean(loss1)#this mean operation is a diluting factor!!!
        #log.info('loss1scalar:%f'%(loss1.item()))#sanity check passed!
        loss2=torch.sum(loss2)/count#torch.mean(loss2)#I should mean over all the negative label samples!
        #if targets<0:#you meet the unsafe point!
            #loss1 =torch.nn.functional.relu(cbfold-targets)#1000000*self.loss_func(logits, targets)#+jacobian(self.forward)-#1000000 for reacher
            #loss2 =torch.nn.functional.relu(cbfnew-targets)#
        #elif targets>=0:
            #loss1=0*torch.nn.functional.relu(targets-cbfold)#don't update this!
            #loss2=0*torch.nn.functional.relu(targets-cbfnew)#just for consistency!
        normdiffthres=(self.gammasafe+self.gammaunsafe)/self.stepstohell#15#I PICK IT TO BE 15#0.05#?
        #loss4=torch.nn.functional.relu(torch.abs(cbfnew-cbfold)-normdiffthres)#
        loss41=torch.nn.functional.relu(cbfnew-cbfold)#I want cbfnew<cbfold in this case
        loss42=torch.nn.functional.relu(cbfold-cbfnew-normdiffthres)#I want cbfnew<cbfold not too much!
        loss43=torch.nn.functional.relu(torch.abs(cbfnew-cbfold)-normdiffthres)#I want the difference of the 2 not too much!
        #think about loss 4 in this case carefully!
        loss4=torch.where(targets<self.gammasafe,loss41+loss42,loss43)#if targets<0, then this means that you have a violation and then you should use loss41+42
        loss4=torch.mean(loss4)#if targets>0, then you not yet encounter a violation, then you just need to use loss43
        #print('next_obs.shape',next_obs.shape)
        loss5=0*loss4
        loss=self.w6*loss1+self.w7*loss2+self.w8*loss4+self.w5*loss5##
        data = {
            'cbf_total': loss.item(),
            'old_safe':self.m10,#want to show the log plots!#0,#
            'new_safe':self.m10,#0,#
            'old_unsafe': max(self.w6*loss1.item(),self.m10),#old safe
            'new_unsafe': max(self.w7*loss2.item(),self.m10),
            'make_it_a_cbf':self.m10,#want to show the log plots!#0,#0,#-0.001,#just for consistency in plotting!
            'closeness_safe':self.m10,
            'closeness_unsafe':max(self.w8*loss4.item(),self.m10),#for plotting!#
            'regularization':self.w5*loss5.item()}
        return loss,data

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
