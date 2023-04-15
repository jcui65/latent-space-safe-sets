"""
Code inspired by https://github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py
                 https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/MPC.py
                 https://github.com/kchua/handful-of-trials/blob/master/dmbrl/misc/optimizers/cem.py
"""

from .policy import Policy

import latentsafesets.utils.pytorch_utils as ptu
import latentsafesets.utils.spb_utils as spbu
from latentsafesets.modules import VanillaVAE, PETSDynamics, ValueFunction, ConstraintEstimator, \
    GoalIndicator, CBFdotEstimator,CBFdotEstimatorlatentplana, VanillaVAE2, PETSDynamics2#they are all there

import torch
import numpy as np
import gym

import logging

log = logging.getLogger('cem')


class CEMSafeSetPolicy(Policy):
    def __init__(self, env: gym.Env,
                 encoder: VanillaVAE,
                 safe_set,
                 value_function: ValueFunction,
                 dynamics_model: PETSDynamics,
                 constraint_function: ConstraintEstimator,
                 goal_indicator: GoalIndicator,
                 cbfdot_function: CBFdotEstimatorlatentplana,#CBFdotEstimator,
                 #encoder2: VanillaVAE2,
                 #dynamics_model2: PETSDynamics2,#,#
                 params):
        log.info("setting up safe set and dynamics model")

        self.env = env
        self.encoder = encoder
        #self.encoder2 = encoder2
        self.safe_set = safe_set#safe set estimator
        self.dynamics_model = dynamics_model
        self.value_function = value_function
        self.constraint_function = constraint_function
        self.goal_indicator = goal_indicator
        self.cbfdot_function = cbfdot_function
        #self.dynamics_model2 = dynamics_model2
        self.logdir = params['logdir']

        self.d_act = params['d_act']#2
        self.d_obs = params['d_obs']#dimension of observation (3,64,64) for spb, (3,3,64,64) for reacher#
        self.d_latent = params['d_latent']#32
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.plan_hor = params['plan_hor']#H=5
        self.random_percent = params['random_percent']#1 in spb/reacher/pushing
        self.popsize = params['num_candidates']#1000
        self.num_elites = params['num_elites']#100
        self.max_iters = params['max_iters']#5
        self.safe_set_thresh = params['safe_set_thresh']#0.8
        self.safe_set_thresh_mult = params['safe_set_thresh_mult']#0.8
        self.safe_set_thresh_mult_iters = params['safe_set_thresh_mult_iters']#5

        self.cbfd_thresh=params['cbfdot_thresh']
        self.cbfd_thresh_mult = params['cbfdot_thresh_mult']  # 0.8

        self.constraint_thresh = params['constr_thresh']#0.2
        self.goal_thresh = params['gi_thresh']#0.5
        self.ignore_safe_set = params['safe_set_ignore']#True#False#False, for ablation study!#changed to true after using cbf dot
        self.ignore_constraints = params['constr_ignore']#True#False#false
        self.ignore_cbfdots = params['cbfd_ignore']  #False#True#True# false
        self.n_particles=params['n_particles']
        self.light=params['light']
        if self.light=='light':
            self.ignore_safe_set = True#False#False, for ablation study!#changed to true after using cbf dot
            self.ignore_constraints = True#False#false
            self.ignore_cbfdots = False
        elif self.light=='ls3':
            self.ignore_safe_set = False#False, for ablation study!#changed to true after using cbf dot
            self.ignore_constraints = False#false
            self.ignore_cbfdots = True
        elif self.light=='nosafety':
            self.ignore_safe_set = True#False, for ablation study!#changed to true after using cbf dot
            self.ignore_constraints = True#false
            self.ignore_cbfdots = True
        self.mean = torch.zeros(self.d_act)#the dimension of action
        self.std = torch.ones(self.d_act)#all the action candidates start from standard normal distribution
        self.ac_buf = np.array([]).reshape(0, self.d_act)#action buffer?
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])#how it is being used?
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])#how it is being used?
        self.action_type=params['action_type']
        self.reduce_horizon=params['reduce_horizon']
        #self.reward_type=params['reward_type']
        #self.conservative=params['conservative']
    @torch.no_grad()
    def act(self, obs):#if using cbf, see the function actcbfd later on
        """
        Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation. Cannot be a batch

        Returns: An action (and possibly the predicted cost)
        """

        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!

        itr = 0#start from 0
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initally 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)#uniformly random from last iter ation
                    action_samples_random = self._sample_actions_random(num_random)#completely random within the action limit!
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where num_constraint_satisfying=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default, and it is a scalar
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        return self.env.action_space.sample()#really random action!NO MPC is implemented!

                    itr = 0#let's start over with itr=0 in this case!#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None#it may stay at zeroth iteration for many iterations!
                    continue

                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]#get those elite trajectories
                #print('elites.shape',elites.shape)#once it is torch.Size([1, 5, 2]), it's gone!
                #print('elites',elites)

                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)#you get not none self.mean and self.std, so that it would be a good starting point for the next iteration!
                # print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!
                #print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:#this means that there is only one trajectory
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)#
                    #print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?

                action_samples = self._sample_actions_normal(self.mean, self.std)
                #print('action_samples', action_samples)#it becomes nan!

            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                #print('emb.shape',emb.shape)# torch.Size([1, 32])
                #print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials

                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                #line 7 in algorithm 1 in the PETS paper!
                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#call forward#each in the model
                    #print(constraints_all.shape)#torch.Size([20, 1000, 5, 1])
                    maxconstraintall=torch.max(constraints_all, dim=0)#value reduce the dimension to [1000,5,1], also contains 2 things, values and indices
                    maxconstraintallvalues=maxconstraintall[0]#the value of maxconstraintall, has dimension [1000,5,1]
                    constraint_viols = torch.sum(maxconstraintallvalues > self.constraint_thresh, dim=1)#those that violate the constraints,shape[1000,1]
                    #tempe=torch.max(constraints_all, dim=0)#a tuple of (value, indices)#let the most dangerous particle safe, then everyone else's safe
                    #print('tempe',tempe.shape)  # val's shape is torch.Size([1000, 5, 1])#thus each element in constraint_viols is between 0 and planning horizon
                    #print((torch.max(constraints_all, dim=0)[0]).shape)#torch.Size([1000, 5, 1])
                    #print((torch.max(constraints_all, dim=0)[0] > self.constraint_thresh).shape)#torch.Size([1000, 5, 1])
                    #print('constraint_viols.shape',constraint_viols.shape)#sum will consume the 1st dim (5 in this case)#1000 0,1,2,3,4,5s
                else:
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!

                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction
                    safe_set_viols = torch.mean(safe_set_all#not max this time
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#f_G in the paper(1000,1)
                #maybe the self.goal_thresh is a bug source?
                values = values + (constraint_viols + safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()

            itr += 1#CEM Evolution method

        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy()

    def actzero(self, obs):#if using cbf, see the function actcbfd later on
        """
        Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation. Cannot be a batch

        Returns: An action (and possibly the predicted cost)
        """

        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        itr = 0#start from 0
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initally 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)#uniformly random from last iter ation
                    action_samples_random = self._sample_actions_random(num_random)#completely random within the action limit!
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where num_constraint_satisfying=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default, and it is a scalar
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        return 0*self.env.action_space.sample()#really 0 action!NO MPC is implemented!
                    itr = 0#let's start over with itr=0 in this case!#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None#it may stay at zeroth iteration for many iterations!
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]#get those elite trajectories
                #print('elites.shape',elites.shape)#once it is torch.Size([1, 5, 2]), it's gone!
                #print('elites',elites)
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)#you get not none self.mean and self.std, so that it would be a good starting point for the next iteration!
                # print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!
                #print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:#this means that there is only one trajectory
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)#
                    #print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                action_samples = self._sample_actions_normal(self.mean, self.std)
                #print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                #print('emb.shape',emb.shape)# torch.Size([1, 32])
                #print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                #line 7 in algorithm 1 in the PETS paper!
                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#call forward#each in the model
                    #print(constraints_all.shape)#torch.Size([20, 1000, 5, 1])
                    maxconstraintall=torch.max(constraints_all, dim=0)#value reduce the dimension to [1000,5,1], also contains 2 things, values and indices
                    maxconstraintallvalues=maxconstraintall[0]#the value of maxconstraintall, has dimension [1000,5,1]
                    constraint_viols = torch.sum(maxconstraintallvalues > self.constraint_thresh, dim=1)#those that violate the constraints,shape[1000,1]
                    #tempe=torch.max(constraints_all, dim=0)#a tuple of (value, indices)#let the most dangerous particle safe, then everyone else's safe
                    #print('tempe',tempe.shape)  # val's shape is torch.Size([1000, 5, 1])#thus each element in constraint_viols is between 0 and planning horizon
                    #print((torch.max(constraints_all, dim=0)[0]).shape)#torch.Size([1000, 5, 1])
                    #print((torch.max(constraints_all, dim=0)[0] > self.constraint_thresh).shape)#torch.Size([1000, 5, 1])
                    #print('constraint_viols.shape',constraint_viols.shape)#sum will consume the 1st dim (5 in this case)#1000 0,1,2,3,4,5s
                else:
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction
                    safe_set_viols = torch.mean(safe_set_all#not max this time
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#f_G in the paper(1000,1)
                #maybe the self.goal_thresh is a bug source?
                values = values + (constraint_viols + safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy()

    @torch.no_grad()
    def actcbfd(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation. Cannot be a batch

        Returns: An action (and possibly the predicted cost)
        """

        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!

        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation
                #print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        return self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!

                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue

                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                #print('elites.shape',elites.shape)#once it is torch.Size([1, 5, 2]), it's gone!
                #print('elites',elites)

                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)
                # print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!
                #print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)#
                    #print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)
                #print('action_samples', action_samples)#it becomes nan!

            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                #print('emb.shape',emb.shape)# torch.Size([1, 32])
                #print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials

                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20

                #print(state.shape)#(2,)
                #print(state.dtype)#float64
                storch=ptu.torchify(state)#state torch
                #print(action_samples.shape)#torch.Size([1000, 5, 2])
                #print(action_samples.dtype)#torch.float32
                se=storch+action_samples#se means state estimated#shape(1000,5,2)
                #se1=stateevolve
                xmove=0#-25#30#
                ymove=-40#0#-30#
                walls=[((75+xmove,45+ymove),(100+xmove,105+ymove))]#[((75+xmove,55+ymove),(100+xmove,95+ymove))]#
                #I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0]<=walls[0][0][0])*(se[:, :, 1]<=walls[0][0][1]), se[:, :, 0]-walls[0][0][0], se[:, :, 0])
                #Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                #and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1=torch.concat((rd1h.reshape(rd1h.shape[0],rd1h.shape[1],1),rd1v.reshape(rd1v.shape[0],rd1v.shape[1],1)),dim=2)
                #we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0])*(rd1[:, :, 0] <= walls[0][1][0]) * (rd1[:, :, 1] <= walls[0][0][1]),
                                   0*rd1[:, :, 0] , rd1[:, :, 0])#region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0])*(rd1[:, :, 0] <= walls[0][1][0]) * (rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition=(rd2[:, :, 0]>walls[0][1][0])*(rd2[:, :, 1]<=walls[0][0][1])#this condition is to see if it is in region 3
                rd3h=torch.where(rd3condition,rd2[:, :, 0]-walls[0][1][0], rd2[:, :, 0])#h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])#v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1])*(rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0*rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1]- walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) *(rd5[:, :, 0] > walls[0][0][0]) * (rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0*rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0]- walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) *(rd7[:, :, 1] <= walls[0][1][1])* (rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0*rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8 = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)#dim: (1000,5,2)
                rdn=torch.norm(rd8,dim=2)#rdn for relative distance norm
                rdnv=rdn<15#rdnv for rdn violator
                rdnvi=torch.sum(rdnv,dim=1)#rdn violator indices
                #print('rdnvi', rdnvi)
                rdnvi=rdnvi.reshape(rdnvi.shape[0],1)

                rdnvc = rdn < 10  # rdnv for rdn violator critical
                rdnvci = torch.sum(rdnvc, dim=1)  # rdn violator critical indices
                #print('rdnvci', rdnvci)
                rdnvci = rdnvci.reshape(rdnvi.shape[0], 1)

                #print(rdn.shape)#torch.Size([1000, 5])
                cbf=rdn**2-15**2#13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                acbf=-cbf*act_cbfd_thresh#acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                asrv1=action_samples[:,:,0]#asrv1 means action sample reversed in the 1st dimension (horizontal dimension)!
                asrv2=action_samples[:,:,1]#-action_samples[:,:,1]#asrv2 means action sample reversed in the 2st dimension (vertical dimension)!
                asrv = torch.concat((asrv1.reshape(asrv1.shape[0], asrv1.shape[1], 1), asrv2.reshape(asrv2.shape[0], asrv2.shape[1], 1)),dim=2)  # dim: (1000,5,2)
                rda=torch.concat((rd8,asrv),dim=2)#check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network


                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!

                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    cbfdots_all = self.cbfdot_function(rda, already_embedded=True)#all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    #print(cbfdots_all.shape)#torch.Size([1000, 5, 1])#
                    cbfdots_all=cbfdots_all.reshape(cbfdots_all.shape[0],cbfdots_all.shape[1])#
                    #print('cbfdots_all', cbfdots_all)
                    cbfdots_viols = torch.sum(cbfdots_all<acbf, dim=1)#those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('acbf',acbf)#bigger than or equal to is the right thing to do! The violations are <!
                    #print('cbfdots_viols',cbfdots_viols)
                    #print('cbfdots_viols', cbfdots_viols)
                    cbfdots_viols=cbfdots_viols.reshape(cbfdots_viols.shape[0],1)#the threshold now should be predictions dependent
                    #print('cbfdots_viols.shape',cbfdots_viols.shape)
                    #print('cbfdots_viols',cbfdots_viols)
                    #print(cbfdots.shape)
                else:#if ignoring the cbf dot constraints
                    cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #print(state)
                #print(state[0])
                #print(state[1])
                if torch.max(rdnvi)>0 or torch.max(cbfdots_viols)>0:#
                    #print('rdnvi>0!')
                    #print('rdnvi', rdnvi.reshape(rdnvi.shape[0]))
                    #print('rdnvi-cbfdots_viols',(rdnvi-cbfdots_viols).reshape(rdnvi.shape[0]))
                    rdnvimask=rdnvi>0.5
                    cbfdots_violsmask=cbfdots_viols>0.5
                    rdnvnotimask = rdnvi < 0.5
                    cbfdots_notviolsmask = cbfdots_viols < 0.5
                    tpmask=rdnvimask*cbfdots_violsmask
                    fpmask=rdnvnotimask*cbfdots_violsmask
                    fnmask=rdnvimask*cbfdots_notviolsmask
                    tnmask=rdnvnotimask*cbfdots_notviolsmask
                    tpcount=torch.sum(tpmask)
                    fpcount=torch.sum(fpmask)
                    fncount=torch.sum(fnmask)
                    tncount=torch.sum(tnmask)
                    tp+=tpcount
                    fp+=fpcount
                    fn+=fncount
                    tn+=tncount
                    #print('tp,fp,fn,tn', tp.item(), fp.item(), fn.item(), tn.item())
                    #log.info(
                        #'tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d' % (tp, fp, fn, tn, tpc, fpc, fnc, tnc))
                    #if torch.max(rdnvci > 0):
                    #print('really critical!')
                    #print('rdnvci-cbfdots_viols', (rdnvci - cbfdots_viols).reshape(rdnvi.shape[0]))
                    #print('really critical done this time!')
                    rdnvcimask = rdnvci > 0.5
                    rdnvnotcimask = rdnvci < 0.5
                    tpcmask = rdnvcimask * cbfdots_violsmask
                    fpcmask = rdnvnotcimask * cbfdots_violsmask
                    fncmask = rdnvcimask * cbfdots_notviolsmask
                    tncmask = rdnvnotcimask * cbfdots_notviolsmask
                    tpccount = torch.sum(tpcmask)
                    fpccount = torch.sum(fpcmask)
                    fnccount = torch.sum(fncmask)
                    tnccount = torch.sum(tncmask)
                    tpc += tpccount
                    fpc += fpccount
                    fnc += fnccount
                    tnc += tnccount
                    #print('tpc,fpc,fnc,tnc', tpc.item(), fpc.item(), fnc.item(), tnc.item())
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                    #else:

                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvi.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvci.shape[0]


                #elif torch.max(cbfdots_viols)>0:#
                    #print('the cbf dot estimator is too sensitive!')
                    #print('rdnvi-cbfdots_viols', (rdnvi - cbfdots_viols).reshape(rdnvi.shape[0]))

                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                #maybe the self.goal_thresh is a bug source?
                values = values + (constraint_viols +cbfdots_viols+ safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5

            itr += 1#CEM Evolution method

        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdcircle(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation. Cannot be a batch

        Returns: An action (and possibly the predicted cost)
        """

        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!

        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation
                #print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        return self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!

                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue

                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                #print('elites.shape',elites.shape)#once it is torch.Size([1, 5, 2]), it's gone!
                #print('elites',elites)

                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)
                # print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!
                #print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)#
                    #print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)
                #print('action_samples', action_samples)#it becomes nan!

            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                #print('emb.shape',emb.shape)# torch.Size([1, 32])
                #print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials

                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20

                #print(state.shape)#(2,)
                #print(state.dtype)#float64
                storch=ptu.torchify(state)#state torch
                #print(action_samples.shape)#torch.Size([1000, 5, 2])
                #print(action_samples.dtype)#torch.float32
                se=storch+action_samples#se means state estimated#shape(1000,5,2)
                #se1=stateevolve
                xmove=0#-25#30#
                ymove=0#-40#-30#
                #walls=[((75+xmove,45+ymove),(100+xmove,105+ymove))]#[((75+xmove,55+ymove),(100+xmove,95+ymove))]#
                #I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                device=se.device
                circlecenter=torch.tensor([90,75]).to(device)
                circleradius=30#25#
                rd=se-torch.asarray(circlecenter)#relative distance vector
                #print('rd',rd)
                rdtan2=torch.atan2(rd[:,:,1],rd[:,:,0])#get the angle
                rdy=circleradius*torch.sin(rdtan2)#relative distance in y direction led by the circular obstacle
                rdx=circleradius*torch.cos(rdtan2)
                rdr = torch.concat(
                    (rdx.reshape(rdx.shape[0], rdx.shape[1], 1), rdy.reshape(rdy.shape[0], rdy.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)#rdr means relative distance induced by the radius of the circular obstacle
                #print('rdr',rdr)
                rd8=rd-rdr
                #print('rd8',rd8)
                rdn=torch.norm(rd8,dim=2)#rdn for relative distance norm
                #print('rdn',rdn)
                rdnv=rdn<15#15=10+5#rdnv for rdn violator
                rdnvi=torch.sum(rdnv,dim=1)#rdn violator indices
                #print('rdnvi', rdnvi)
                rdnvi=rdnvi.reshape(rdnvi.shape[0],1)

                rdnvc = rdn < 10  # rdnv for rdn violator critical
                rdnvci = torch.sum(rdnvc, dim=1)  # rdn violator critical indices
                #print('rdnvci', rdnvci)
                rdnvci = rdnvci.reshape(rdnvi.shape[0], 1)

                #print(rdn.shape)#torch.Size([1000, 5])
                cbf=rdn**2-15**2#13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                #print('cbf',cbf)
                acbf=-cbf*act_cbfd_thresh#acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                asrv1=action_samples[:,:,0]#asrv1 means action sample reversed in the 1st dimension (horizontal dimension)!
                asrv2=action_samples[:,:,1]#-action_samples[:,:,1]#asrv2 means action sample reversed in the 2st dimension (vertical dimension)!
                asrv = torch.concat((asrv1.reshape(asrv1.shape[0], asrv1.shape[1], 1), asrv2.reshape(asrv2.shape[0], asrv2.shape[1], 1)),dim=2)  # dim: (1000,5,2)
                rda=torch.concat((rd8,asrv),dim=2)#check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network


                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!

                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    cbfdots_all = self.cbfdot_function(rda, already_embedded=True)#all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    #print(cbfdots_all.shape)#torch.Size([1000, 5, 1])#
                    cbfdots_all=cbfdots_all.reshape(cbfdots_all.shape[0],cbfdots_all.shape[1])#
                    #print('cbfdots_all', cbfdots_all)
                    cbfdots_viols = torch.sum(cbfdots_all<acbf, dim=1)#those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('acbf',acbf)#bigger than or equal to is the right thing to do! The violations are <!
                    #print('cbfdots_viols',cbfdots_viols)
                    #print('cbfdots_viols', cbfdots_viols)
                    cbfdots_viols=cbfdots_viols.reshape(cbfdots_viols.shape[0],1)#the threshold now should be predictions dependent
                    #print('cbfdots_viols.shape',cbfdots_viols.shape)
                    #print('cbfdots_viols',cbfdots_viols)
                    #print(cbfdots.shape)
                else:#if ignoring the cbf dot constraints
                    cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #print(state)
                #print(state[0])
                #print(state[1])
                if torch.max(rdnvi)>0 or torch.max(cbfdots_viols)>0:#
                    #print('rdnvi>0!')
                    #print('rdnvi', rdnvi.reshape(rdnvi.shape[0]))
                    #print('rdnvi-cbfdots_viols',(rdnvi-cbfdots_viols).reshape(rdnvi.shape[0]))
                    rdnvimask=rdnvi>0.5
                    cbfdots_violsmask=cbfdots_viols>0.5
                    rdnvnotimask = rdnvi < 0.5
                    cbfdots_notviolsmask = cbfdots_viols < 0.5
                    tpmask=rdnvimask*cbfdots_violsmask
                    fpmask=rdnvnotimask*cbfdots_violsmask
                    fnmask=rdnvimask*cbfdots_notviolsmask
                    tnmask=rdnvnotimask*cbfdots_notviolsmask
                    tpcount=torch.sum(tpmask)
                    fpcount=torch.sum(fpmask)
                    fncount=torch.sum(fnmask)
                    tncount=torch.sum(tnmask)
                    tp+=tpcount
                    fp+=fpcount
                    fn+=fncount
                    tn+=tncount
                    #print('tp,fp,fn,tn', tp.item(), fp.item(), fn.item(), tn.item())
                    #log.info(
                        #'tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d' % (tp, fp, fn, tn, tpc, fpc, fnc, tnc))
                    #if torch.max(rdnvci > 0):
                    #print('really critical!')
                    #print('rdnvci-cbfdots_viols', (rdnvci - cbfdots_viols).reshape(rdnvi.shape[0]))
                    #print('really critical done this time!')
                    rdnvcimask = rdnvci > 0.5
                    rdnvnotcimask = rdnvci < 0.5
                    tpcmask = rdnvcimask * cbfdots_violsmask
                    fpcmask = rdnvnotcimask * cbfdots_violsmask
                    fncmask = rdnvcimask * cbfdots_notviolsmask
                    tncmask = rdnvnotcimask * cbfdots_notviolsmask
                    tpccount = torch.sum(tpcmask)
                    fpccount = torch.sum(fpcmask)
                    fnccount = torch.sum(fncmask)
                    tnccount = torch.sum(tncmask)
                    tpc += tpccount
                    fpc += fpccount
                    fnc += fnccount
                    tnc += tnccount
                    #print('tpc,fpc,fnc,tnc', tpc.item(), fpc.item(), fnc.item(), tnc.item())
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                    #else:

                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvi.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvci.shape[0]


                #elif torch.max(cbfdots_viols)>0:#
                    #print('the cbf dot estimator is too sensitive!')
                    #print('rdnvi-cbfdots_viols', (rdnvi - cbfdots_viols).reshape(rdnvi.shape[0]))

                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                #maybe the self.goal_thresh is a bug source?
                values = values + (constraint_viols +cbfdots_viols+ safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5

            itr += 1#CEM Evolution method

        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarecircle(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation. Cannot be a batch

        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1)#
        #print('embrepeat.shape',embrepeat.shape)
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation
                #print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        return self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!

                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)
                # print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!
                #print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)#
                    #print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)
                #print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                storch=ptu.torchify(state)#state torch
                se=storch+action_samples#se means state estimated#shape(1000,5,2)
                #se1=stateevolve

                #the square part
                xmove=0#-25#30#
                ymove=0#-45#-40#-35#-33#-30#-25#
                lux = 75#50
                luy = 55
                width = 25#20#
                height = 40# 50#
                walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))]  #
                # I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 0] - walls[0][0][0], se[:, :, 0])
                # Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                # and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1 = torch.concat(
                    (rd1h.reshape(rd1h.shape[0], rd1h.shape[1], 1), rd1v.reshape(rd1v.shape[0], rd1v.shape[1], 1)),
                    dim=2)
                # we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   0 * rd1[:, :, 0], rd1[:, :, 0])  # region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition = (rd2[:, :, 0] > walls[0][1][0]) * (
                            rd2[:, :, 1] <= walls[0][0][1])  # this condition is to see if it is in region 3
                rd3h = torch.where(rd3condition, rd2[:, :, 0] - walls[0][1][0], rd2[:, :, 0])  # h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])  # v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1]) * (
                            rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0 * rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1] - walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) * (rd5[:, :, 0] > walls[0][0][0]) * (
                            rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0 * rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0] - walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) * (rd7[:, :, 1] <= walls[0][1][1]) * (
                            rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0 * rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8s = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)
                rdns = torch.norm(rd8s, dim=2)  # rdn for relative distance norm
                rdnvs = rdns < 15  # rdnv for rdn violator
                rdnvis = torch.sum(rdnvs, dim=1)  # rdn violator indices# print('rdnvi', rdnvi)
                rdnvis = rdnvis.reshape(rdnvis.shape[0], 1)
                rdnvcs = rdns < 10  # rdnv for rdn violator critical
                rdnvcis = torch.sum(rdnvcs, dim=1)  # rdn violator critical indices# print('rdnvci', rdnvci)
                rdnvcis = rdnvcis.reshape(rdnvis.shape[0], 1)# print(rdn.shape)#torch.Size([1000, 5])
                cbfs = rdns ** 2 - 15 ** 2  # 13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                acbfs = -cbfs * act_cbfd_thresh  # acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                #rdas = torch.concat((rd8s, action_samples),
                                   #dim=2)  # check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network
                #print('action_samples.shape',action_samples.shape)
                #rdas=torch.concat((embrepeat, action_samples),dim=2)
                #the circle part
                device=se.device
                centerx=115#118#
                centery=85#80#75#
                circlecenter=torch.tensor([centerx,centery]).to(device)
                circleradius=20#19#15#14#30#25#
                rdc=se-torch.asarray(circlecenter)#relative distance vector#print('rd',rd)
                rdtan2c=torch.atan2(rdc[:,:,1],rdc[:,:,0])#get the angle
                rdyc=circleradius*torch.sin(rdtan2c)#relative distance in y direction led by the circular obstacle
                rdxc=circleradius*torch.cos(rdtan2c)
                rdrc = torch.concat(
                    (rdxc.reshape(rdxc.shape[0], rdxc.shape[1], 1), rdyc.reshape(rdyc.shape[0], rdyc.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)#rdr means relative distance induced by the radius of the circular obstacle
                #print('rdr',rdr)
                rd8c=rdc-rdrc#print('rd8',rd8)
                rdnc=torch.norm(rd8c,dim=2)#rdn for relative distance norm#print('rdn',rdn)
                rdnvc=rdnc<15#15=10+5#rdnv for rdn violator
                rdnvic=torch.sum(rdnvc,dim=1)#rdn violator indices#print('rdnvi', rdnvi)
                rdnvic=rdnvic.reshape(rdnvic.shape[0],1)
                rdnvcc = rdnc < 10  # rdnv for rdn violator critical
                rdnvcic = torch.sum(rdnvcc, dim=1)  # rdn violator critical indices#print('rdnvci', rdnvci)
                rdnvcic = rdnvcic.reshape(rdnvic.shape[0], 1)
                cbfc=rdnc**2-15**2#13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                #print('cbf',cbf)
                acbfc=-cbfc*act_cbfd_thresh#acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                #rdac=torch.concat((rd8c,action_samples),dim=2)#check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network
                #rdac = torch.concat((embrepeat, action_samples), dim=2)
                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbfdots_alls = self.cbfdot_function(embrepeat,action_samples,already_embedded=True) #with the reformulated cbfd estimator
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1])  #
                    cbfdots_violss = torch.sum(cbfdots_alls < acbfs,
                                               dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                    #cbfdots_allc = self.cbfdot_function(rdac, already_embedded=True)#all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbfdots_allc = self.cbfdot_function(embrepeat, action_samples,already_embedded=True)  # with the reformulated cbfd estimator
                    cbfdots_allc=cbfdots_allc.reshape(cbfdots_allc.shape[0],cbfdots_allc.shape[1])#
                    cbfdots_violsc = torch.sum(cbfdots_allc<acbfc, dim=1)#those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbfdots_violsc=cbfdots_violsc.reshape(cbfdots_violsc.shape[0],1)#the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints
                    cbfdots_violsc = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                    rdnvimasks = rdnvis > 0.5
                    cbfdots_violsmasks = cbfdots_violss > 0.5
                    rdnvnotimasks = rdnvis< 0.5
                    cbfdots_notviolsmasks = cbfdots_violss < 0.5
                    tpmasks = rdnvimasks * cbfdots_violsmasks
                    fpmasks = rdnvnotimasks * cbfdots_violsmasks
                    fnmasks = rdnvimasks * cbfdots_notviolsmasks
                    tnmasks = rdnvnotimasks * cbfdots_notviolsmasks
                    tpcounts = torch.sum(tpmasks)
                    fpcounts = torch.sum(fpmasks)
                    fncounts = torch.sum(fnmasks)
                    tncounts = torch.sum(tnmasks)
                    tp += tpcounts
                    fp += fpcounts
                    fn += fncounts
                    tn += tncounts

                    rdnvimaskc=rdnvic>0.5
                    cbfdots_violsmaskc=cbfdots_violsc>0.5
                    rdnvnotimaskc = rdnvic < 0.5
                    cbfdots_notviolsmaskc = cbfdots_violsc < 0.5
                    tpmaskc=rdnvimaskc*cbfdots_violsmaskc
                    fpmaskc=rdnvnotimaskc*cbfdots_violsmaskc
                    fnmaskc=rdnvimaskc*cbfdots_notviolsmaskc
                    tnmaskc=rdnvnotimaskc*cbfdots_notviolsmaskc
                    tpcountc=torch.sum(tpmaskc)
                    fpcountc=torch.sum(fpmaskc)
                    fncountc=torch.sum(fnmaskc)
                    tncountc=torch.sum(tnmaskc)
                    tp+=tpcountc
                    fp+=fpcountc
                    fn+=fncountc
                    tn+=tncountc

                    rdnvcicmasks = rdnvcis > 0.5
                    rdnvnotcimasks = rdnvcis < 0.5
                    tpcmasks = rdnvcicmasks * cbfdots_violsmasks
                    fpcmasks = rdnvnotcimasks * cbfdots_violsmasks
                    fncmasks = rdnvcicmasks * cbfdots_notviolsmasks
                    tncmasks = rdnvnotcimasks * cbfdots_notviolsmasks
                    tpccounts = torch.sum(tpcmasks)
                    fpccounts = torch.sum(fpcmasks)
                    fnccounts = torch.sum(fncmasks)
                    tnccounts = torch.sum(tncmasks)
                    tpc += tpccounts
                    fpc += fpccounts
                    fnc += fnccounts
                    tnc += tnccounts

                    rdnvcicmaskc = rdnvcic > 0.5
                    rdnvnotcimaskc = rdnvcic < 0.5
                    tpcmaskc = rdnvcicmaskc * cbfdots_violsmaskc
                    fpcmaskc = rdnvnotcimaskc * cbfdots_violsmaskc
                    fncmaskc = rdnvcicmaskc * cbfdots_notviolsmaskc
                    tncmaskc = rdnvnotcimaskc * cbfdots_notviolsmaskc
                    tpccountc = torch.sum(tpcmaskc)
                    fpccountc = torch.sum(fpcmaskc)
                    fnccountc = torch.sum(fncmaskc)
                    tnccountc = torch.sum(tncmaskc)
                    tpc += tpccountc
                    fpc += fpccountc
                    fnc += fnccountc
                    tnc += tnccountc#print('tpc,fpc,fnc,tnc', tpc.item(), fpc.item(), fnc.item(), tnc.item())
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvic.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvcic.shape[0]
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+cbfdots_violsc+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatent(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        return self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                storch=ptu.torchify(state)#state torch
                se=storch+action_samples#se means state estimated#shape(1000,5,2)#se1=stateevolve
                #the square part
                xmove=0#-25#30#
                ymove=0#-45#-40#-35#-33#-30#-25#
                lux = 105#75#50#
                luy = 55#40#
                width = 25#20#
                height = 40# 50#
                walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))]  #
                # I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 0] - walls[0][0][0], se[:, :, 0])
                # Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                # and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1 = torch.concat(
                    (rd1h.reshape(rd1h.shape[0], rd1h.shape[1], 1), rd1v.reshape(rd1v.shape[0], rd1v.shape[1], 1)),
                    dim=2)
                # we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   0 * rd1[:, :, 0], rd1[:, :, 0])  # region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition = (rd2[:, :, 0] > walls[0][1][0]) * (
                            rd2[:, :, 1] <= walls[0][0][1])  # this condition is to see if it is in region 3
                rd3h = torch.where(rd3condition, rd2[:, :, 0] - walls[0][1][0], rd2[:, :, 0])  # h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])  # v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1]) * (
                            rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0 * rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1] - walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) * (rd5[:, :, 0] > walls[0][0][0]) * (
                            rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0 * rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0] - walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) * (rd7[:, :, 1] <= walls[0][1][1]) * (
                            rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0 * rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8s = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)
                rdns = torch.norm(rd8s, dim=2)  # rdn for relative distance norm
                rdnvs = rdns < 15  # rdnv for rdn violator
                rdnvis = torch.sum(rdnvs, dim=1)  # rdn violator indices# print('rdnvi', rdnvi)
                rdnvis = rdnvis.reshape(rdnvis.shape[0], 1)
                rdnvcs = rdns < 10  # rdnv for rdn violator critical
                rdnvcis = torch.sum(rdnvcs, dim=1)  # rdn violator critical indices# print('rdnvci', rdnvci)
                rdnvcis = rdnvcis.reshape(rdnvis.shape[0], 1)# print(rdn.shape)#torch.Size([1000, 5])
                cbfs = rdns ** 2 - 15 ** 2  # 13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                acbfs = -cbfs * act_cbfd_thresh  # acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                #rdas = torch.concat((rd8s, action_samples),
                                   #dim=2)  # check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network
                #print('action_samples.shape',action_samples.shape)
                #rdas=torch.concat((embrepeat, action_samples),dim=2)

                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbfdots_alls = self.cbfdot_function(embrepeat,action_samples,already_embedded=True) #with the reformulated cbfd estimator
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1])  #
                    cbfdots_violss = torch.sum(cbfdots_alls < acbfs,
                                               dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                if torch.max(rdnvis) > 0 or torch.max(cbfdots_violss) > 0:  ##
                    rdnvimasks = rdnvis > 0.5
                    cbfdots_violsmasks = cbfdots_violss > 0.5
                    rdnvnotimasks = rdnvis< 0.5
                    cbfdots_notviolsmasks = cbfdots_violss < 0.5
                    tpmasks = rdnvimasks * cbfdots_violsmasks
                    fpmasks = rdnvnotimasks * cbfdots_violsmasks
                    fnmasks = rdnvimasks * cbfdots_notviolsmasks
                    tnmasks = rdnvnotimasks * cbfdots_notviolsmasks
                    tpcounts = torch.sum(tpmasks)
                    fpcounts = torch.sum(fpmasks)
                    fncounts = torch.sum(fnmasks)
                    tncounts = torch.sum(tnmasks)
                    tp += tpcounts
                    fp += fpcounts
                    fn += fncounts
                    tn += tncounts

                    rdnvcicmasks = rdnvcis > 0.5
                    rdnvnotcimasks = rdnvcis < 0.5
                    tpcmasks = rdnvcicmasks * cbfdots_violsmasks
                    fpcmasks = rdnvnotcimasks * cbfdots_violsmasks
                    fncmasks = rdnvcicmasks * cbfdots_notviolsmasks
                    tncmasks = rdnvnotcimasks * cbfdots_notviolsmasks
                    tpccounts = torch.sum(tpcmasks)
                    fpccounts = torch.sum(fpcmasks)
                    fnccounts = torch.sum(fncmasks)
                    tnccounts = torch.sum(tncmasks)
                    tpc += tpccounts
                    fpc += fpccounts
                    fnc += fnccounts
                    tnc += tnccounts
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvis.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvcis.shape[0]
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplana(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        if self.action_type=='zero':
                            return 0*self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                        elif self.action_type=='random':
                            return self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                storch=ptu.torchify(state)#state torch
                se=storch+action_samples#se means state estimated#shape(1000,5,2)#se1=stateevolve
                #the square part
                xmove=0#-25#30#
                ymove=0#-45#-40#-35#-33#-30#-25#
                lux = 75#105#50#
                luy = 55#40#
                width = 25#20#
                height = 40# 50#
                walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))]  #
                # I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 0] - walls[0][0][0], se[:, :, 0])
                # Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                # and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1 = torch.concat(
                    (rd1h.reshape(rd1h.shape[0], rd1h.shape[1], 1), rd1v.reshape(rd1v.shape[0], rd1v.shape[1], 1)),
                    dim=2)
                # we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   0 * rd1[:, :, 0], rd1[:, :, 0])  # region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition = (rd2[:, :, 0] > walls[0][1][0]) * (
                            rd2[:, :, 1] <= walls[0][0][1])  # this condition is to see if it is in region 3
                rd3h = torch.where(rd3condition, rd2[:, :, 0] - walls[0][1][0], rd2[:, :, 0])  # h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])  # v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1]) * (
                            rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0 * rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1] - walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) * (rd5[:, :, 0] > walls[0][0][0]) * (
                            rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0 * rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0] - walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) * (rd7[:, :, 1] <= walls[0][1][1]) * (
                            rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0 * rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8s = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)
                rdns = torch.norm(rd8s, dim=2)  # rdn for relative distance norm
                rdnvs = rdns < 6#5#15  # rdnv for rdn violator
                rdnvis = torch.sum(rdnvs, dim=1)  # rdn violator indices# print('rdnvi', rdnvi)
                rdnvis = rdnvis.reshape(rdnvis.shape[0], 1)
                rdnvcs = rdns < 1e-8#10  # rdnv for rdn violator critical
                rdnvcis = torch.sum(rdnvcs, dim=1)  # rdn violator critical indices# print('rdnvci', rdnvci)
                rdnvcis = rdnvcis.reshape(rdnvis.shape[0], 1)# print(rdn.shape)#torch.Size([1000, 5])
                #cbfs = rdns ** 2 - 15 ** 2  # 13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                #acbfs = -cbfs * act_cbfd_thresh  # acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                #rdas = torch.concat((rd8s, action_samples),
                                   #dim=2)  # check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network
                #print('action_samples.shape',action_samples.shape)
                #rdas=torch.concat((embrepeat, action_samples),dim=2)

                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    #print('cbf_alls',cbf_alls)
                    cbf_alls4=cbf_alls[:,:,0:self.plan_hor-1,:]
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #print('cbfdots_alls', cbfdots_alls)  #
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                                            #dim=1)  # sum over planning horizon#f_G in the paper(1000,1)
                    #cbfdots_violss = torch.sum(cbfdots_alls < acbfs,#the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    #print('cbf_initalls4', cbf_initalls4)  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls,dim=0) < acbfs,  # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls, dim=0) < torch.mean(acbfs,dim=0),
                                               # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements
                    #print('lhse.shape',lhse.shape)
                    rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices
                    #print('rhse.shape', rhse.shape)
                    cbfdots_violss = torch.sum(( lhse< rhse),dim=1) # the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('cbfdots_violss',cbfdots_violss)
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                if torch.max(rdnvis) > 0 or torch.max(cbfdots_violss) > 0:  ##
                    rdnvimasks = rdnvis > 0.5
                    cbfdots_violsmasks = cbfdots_violss > 0.5
                    rdnvnotimasks = rdnvis< 0.5
                    cbfdots_notviolsmasks = cbfdots_violss < 0.5
                    tpmasks = rdnvimasks * cbfdots_violsmasks
                    fpmasks = rdnvnotimasks * cbfdots_violsmasks
                    fnmasks = rdnvimasks * cbfdots_notviolsmasks
                    tnmasks = rdnvnotimasks * cbfdots_notviolsmasks
                    tpcounts = torch.sum(tpmasks)
                    fpcounts = torch.sum(fpmasks)
                    fncounts = torch.sum(fnmasks)
                    tncounts = torch.sum(tnmasks)
                    tp += tpcounts
                    fp += fpcounts
                    fn += fncounts
                    tn += tncounts

                    rdnvcicmasks = rdnvcis > 0.5
                    rdnvnotcimasks = rdnvcis < 0.5
                    tpcmasks = rdnvcicmasks * cbfdots_violsmasks
                    fpcmasks = rdnvnotcimasks * cbfdots_violsmasks
                    fncmasks = rdnvcicmasks * cbfdots_notviolsmasks
                    tncmasks = rdnvnotcimasks * cbfdots_notviolsmasks
                    tpccounts = torch.sum(tpcmasks)
                    fpccounts = torch.sum(fpcmasks)
                    fnccounts = torch.sum(fncmasks)
                    tnccounts = torch.sum(tncmasks)
                    tpc += tpccounts
                    fpc += fpccounts
                    fnc += fnccounts
                    tnc += tnccounts
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvis.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvcis.shape[0]
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 10*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 100*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplanareacher(self, obs):#,state):#,tp,fp,fn,tn,tpc,fpc,fnc,tncsome intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        #embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        #print('env.state',state)
        randflag=0#this is the flag to show if a random action is finally being chosen!
        cbfhorizon=self.plan_hor
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    if self.reduce_horizon=='no':
                        act_cbfd_thresh *= 1  # *0.8 by default
                        #act_cbfd_thresh=min(act_cbfd_thresh,1)
                        log.info('alpha keeps %f'%(act_cbfd_thresh))
                    elif self.reduce_horizon=='alpha':
                        act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                        act_cbfd_thresh=min(act_cbfd_thresh,1)
                        log.info('alpha increased to %f'%(act_cbfd_thresh))
                    elif self.reduce_horizon=='horizon':
                        cbfhorizon-=1
                        cbfhorizon=max(1,cbfhorizon)
                        log.info('horizon reduced to %d'%(cbfhorizon))
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('no trajectory candidates satisfy constraints! The BF is doing its job? Picking random actions!')
                        #log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            #tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        randflag=1
                        if self.action_type=='random':
                            return self.env.action_space.sample(),randflag#for fair comparison#
                        elif self.action_type=='zero':
                            return 0*self.env.action_space.sample(),randflag#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!#
                        elif self.action_type=='recovery':
                            #do CEM just to max the barrier function value
                            itrrecovery = 0#start from 0
                            while itrrecovery < self.max_iters:#5
                                if itrrecovery == 0:
                                    # Action samples dim (num_candidates, planning_hor, d_act)
                                    if self.mean is None:#right after reset
                                        action_samples = self._sample_actions_random()#1000*5 2d array
                                    else:
                                        num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                                        num_dist = self.popsize - num_random#=0 when random_percent=1
                                        action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)#uniformly random from last iter ation
                                        action_samples_random = self._sample_actions_random(num_random)#completely random within the action limit!
                                        action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
                                else:
                                    # Chop off the numer of elites so we don't use constraint violating trajectories
                                    iter_num_elites = self.num_elites#min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                                    #what if I change this into num_constraint_satisfying+2?
                                    # Sort
                                    #sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                                    sortid = cbfs.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                                    actions_sorted = action_samples[sortid]
                                    elites = actions_sorted[-iter_num_elites:]#get those elite trajectories
                                    # Refitting to Best Trajs
                                    self.mean, self.std = elites.mean(0), elites.std(0)#you get not none self.mean and self.std, so that it would be a good starting point for the next iteration!
                                    #import ipdb#it seems that they are lucky to run into the following case

                                    action_samples = self._sample_actions_normal(self.mean, self.std)
                                    #print('action_samples', action_samples)#it becomes nan!

                                if itrrecovery < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                                    # dimension (num_models, num_candidates, planning_hor, d_latent)
                                    #print('emb.shape',emb.shape)# torch.Size([1, 32])#print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                                    predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                                    num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                                    first_states = predictions[:, :, 0, :].reshape((num_models * num_candidates, d_latent))#the 20000*32 comes out!
                                    #last_states = predictions[:, :, -1, :].reshape((num_models * num_candidates, d_latent))#the 20000*32 comes out!
                                    #all_values = self.value_function.get_value(last_states, already_embedded=True)
                                    all_cbfs=self.cbfdot_function(first_states, already_embedded=True)
                                    #nans = torch.isnan(all_values)#should get it from the cbfd function!
                                    #all_values[nans] = -1e5
                                    nanscbf = torch.isnan(all_cbfs)#should get it from the cbfd function!
                                    all_cbfs[nanscbf] = -1e4
                                    #values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                                    cbfs,indices=torch.min(all_cbfs.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                                    #line 7 in algorithm 1 in the PETS paper!#this min may have some robustness effect!
                                    #values = values.squeeze()
                                    cbfs=cbfs.squeeze()
                                itrrecovery += 1#CEM Evolution method
                                #print('itrrecovery',itrrecovery)
                            # Return the best action
                            action = actions_sorted[-1][0]#the best one
                            return action.detach().cpu().numpy(),randflag
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    #print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    eshape=elites.shape
                    log.info('eshape[0]:%d,eshape[1]:%d,eshape[2]:%d' % (eshape[0],eshape[1],eshape[2]))
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                '''
                storch=ptu.torchify(state)#state torch
                se=storch+action_samples#se means state estimated#shape(1000,5,2)#se1=stateevolve
                #the square part
                xmove=0#-25#30#
                ymove=0#-45#-40#-35#-33#-30#-25#
                lux = 75#105#50#
                luy = 55#40#
                width = 25#20#
                height = 40# 50#
                walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))]  #
                # I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 0] - walls[0][0][0], se[:, :, 0])
                # Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                # and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1 = torch.concat(
                    (rd1h.reshape(rd1h.shape[0], rd1h.shape[1], 1), rd1v.reshape(rd1v.shape[0], rd1v.shape[1], 1)),
                    dim=2)
                # we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   0 * rd1[:, :, 0], rd1[:, :, 0])  # region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition = (rd2[:, :, 0] > walls[0][1][0]) * (
                            rd2[:, :, 1] <= walls[0][0][1])  # this condition is to see if it is in region 3
                rd3h = torch.where(rd3condition, rd2[:, :, 0] - walls[0][1][0], rd2[:, :, 0])  # h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])  # v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1]) * (
                            rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0 * rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1] - walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) * (rd5[:, :, 0] > walls[0][0][0]) * (
                            rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0 * rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0] - walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) * (rd7[:, :, 1] <= walls[0][1][1]) * (
                            rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0 * rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8s = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)
                rdns = torch.norm(rd8s, dim=2)  # rdn for relative distance norm
                rdnvs = rdns < 6#5#15  # rdnv for rdn violator
                rdnvis = torch.sum(rdnvs, dim=1)  # rdn violator indices# print('rdnvi', rdnvi)
                rdnvis = rdnvis.reshape(rdnvis.shape[0], 1)
                rdnvcs = rdns < 1e-8#10  # rdnv for rdn violator critical
                rdnvcis = torch.sum(rdnvcs, dim=1)  # rdn violator critical indices# print('rdnvci', rdnvci)
                rdnvcis = rdnvcis.reshape(rdnvis.shape[0], 1)# print(rdn.shape)#torch.Size([1000, 5])
                #cbfs = rdns ** 2 - 15 ** 2  # 13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                #acbfs = -cbfs * act_cbfd_thresh  # acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                #rdas = torch.concat((rd8s, action_samples),
                                   #dim=2)  # check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network
                #print('action_samples.shape',action_samples.shape)
                #rdas=torch.concat((embrepeat, action_samples),dim=2)
                '''
                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    #print('cbf_alls',cbf_alls)
                    cbf_alls4=cbf_alls[:,:,0:cbfhorizon-1,:]#[:,:,0:self.plan_hor-1,:]#
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)#if cbfhorizon-1=0, it should be fine
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls[:,:,0:cbfhorizon,:]-cbf_initalls4#the mean is also subject to change#I think it is correct!
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #print('cbfdots_alls', cbfdots_alls)  #
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                                            #dim=1)  # sum over planning horizon#f_G in the paper(1000,1)
                    #cbfdots_violss = torch.sum(cbfdots_alls < acbfs,#the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    #print('cbf_initalls4', cbf_initalls4)  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #the dim should be consistent
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls,dim=0) < acbfs,  # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls, dim=0) < torch.mean(acbfs,dim=0),
                                               # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements
                    #print('lhse.shape',lhse.shape)
                    rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices
                    #print('rhse.shape', rhse.shape)
                    cbfdots_violss = torch.sum(( lhse< rhse),dim=1) # the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('cbfdots_violss',cbfdots_violss)
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                '''
                if torch.max(rdnvis) > 0 or torch.max(cbfdots_violss) > 0:  ##
                    rdnvimasks = rdnvis > 0.5
                    cbfdots_violsmasks = cbfdots_violss > 0.5
                    rdnvnotimasks = rdnvis< 0.5
                    cbfdots_notviolsmasks = cbfdots_violss < 0.5
                    tpmasks = rdnvimasks * cbfdots_violsmasks
                    fpmasks = rdnvnotimasks * cbfdots_violsmasks
                    fnmasks = rdnvimasks * cbfdots_notviolsmasks
                    tnmasks = rdnvnotimasks * cbfdots_notviolsmasks
                    tpcounts = torch.sum(tpmasks)
                    fpcounts = torch.sum(fpmasks)
                    fncounts = torch.sum(fnmasks)
                    tncounts = torch.sum(tnmasks)
                    tp += tpcounts
                    fp += fpcounts
                    fn += fncounts
                    tn += tncounts

                    rdnvcicmasks = rdnvcis > 0.5
                    rdnvnotcimasks = rdnvcis < 0.5
                    tpcmasks = rdnvcicmasks * cbfdots_violsmasks
                    fpcmasks = rdnvnotcimasks * cbfdots_violsmasks
                    fncmasks = rdnvcicmasks * cbfdots_notviolsmasks
                    tncmasks = rdnvnotcimasks * cbfdots_notviolsmasks
                    tpccounts = torch.sum(tpcmasks)
                    fpccounts = torch.sum(fpcmasks)
                    fnccounts = torch.sum(fncmasks)
                    tnccounts = torch.sum(tncmasks)
                    tpc += tpccounts
                    fpc += fpccounts
                    fnc += fnccounts
                    tnc += tnccounts
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvis.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvcis.shape[0]
                '''
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 10*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 100*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(),randflag#, tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplanareacheraverage(self, obs):#,state):#,tp,fp,fn,tn,tpc,fpc,fnc,tncsome intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        #embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        randflag=0#this is the flag to show if a random action is finally being chosen!
        cbfhorizon=self.plan_hor
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    if self.reduce_horizon=='no':
                        act_cbfd_thresh *= 1  # *0.8 by default
                        #act_cbfd_thresh=min(act_cbfd_thresh,1)
                        log.info('alpha keeps %f'%(act_cbfd_thresh))
                    elif self.reduce_horizon=='alpha':
                        act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                        act_cbfd_thresh=min(act_cbfd_thresh,1)
                        log.info('alpha increased to %f'%(act_cbfd_thresh))
                    elif self.reduce_horizon=='horizon':
                        cbfhorizon-=1
                        cbfhorizon=max(1,cbfhorizon)
                        log.info('horizon reduced to %d'%(cbfhorizon))
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('no trajectory candidates satisfy constraints! The BF is doing its job? Picking random actions!')
                        #log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            #tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        randflag=1
                        if self.action_type=='random':
                            return self.env.action_space.sample(),randflag#for fair comparison#
                        elif self.action_type=='zero':
                            return 0*self.env.action_space.sample(),randflag#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#
                        elif self.action_type=='recovery':
                            #do CEM just to max the barrier function value
                            itrrecovery = 0#start from 0
                            while itrrecovery < self.max_iters:#5
                                if itrrecovery == 0:
                                    # Action samples dim (num_candidates, planning_hor, d_act)
                                    if self.mean is None:#right after reset
                                        action_samples = self._sample_actions_random()#1000*5 2d array
                                    else:
                                        num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                                        num_dist = self.popsize - num_random#=0 when random_percent=1
                                        action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)#uniformly random from last iter ation
                                        action_samples_random = self._sample_actions_random(num_random)#completely random within the action limit!
                                        action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
                                else:
                                    # Chop off the numer of elites so we don't use constraint violating trajectories
                                    iter_num_elites = self.num_elites#min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                                    #what if I change this into num_constraint_satisfying+2?
                                    # Sort
                                    #sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                                    sortid = cbfs.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                                    actions_sorted = action_samples[sortid]
                                    elites = actions_sorted[-iter_num_elites:]#get those elite trajectories
                                    # Refitting to Best Trajs
                                    self.mean, self.std = elites.mean(0), elites.std(0)#you get not none self.mean and self.std, so that it would be a good starting point for the next iteration!
                                    #import ipdb#it seems that they are lucky to run into the following case

                                    action_samples = self._sample_actions_normal(self.mean, self.std)
                                    #print('action_samples', action_samples)#it becomes nan!

                                if itrrecovery < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                                    # dimension (num_models, num_candidates, planning_hor, d_latent)
                                    #print('emb.shape',emb.shape)# torch.Size([1, 32])#print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                                    predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                                    num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                                    first_states = predictions[:, :, 0, :].reshape((num_models * num_candidates, d_latent))#the 20000*32 comes out!
                                    #last_states = predictions[:, :, -1, :].reshape((num_models * num_candidates, d_latent))#the 20000*32 comes out!
                                    #all_values = self.value_function.get_value(last_states, already_embedded=True)
                                    all_cbfs=self.cbfdot_function(first_states, already_embedded=True)
                                    #nans = torch.isnan(all_values)#should get it from the cbfd function!
                                    #all_values[nans] = -1e5
                                    nanscbf = torch.isnan(all_cbfs)#should get it from the cbfd function!
                                    all_cbfs[nanscbf] = -1e4
                                    #values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                                    cbfs,indices=torch.min(all_cbfs.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                                    #line 7 in algorithm 1 in the PETS paper!
                                    #values = values.squeeze()
                                    cbfs=cbfs.squeeze()
                                itrrecovery += 1#CEM Evolution method
                                #print('itrrecovery',itrrecovery)
                            # Return the best action
                            action = actions_sorted[-1][0]#the best one
                            return action.detach().cpu().numpy(),randflag
                        #return self.env.action_space.sample()#for fair comparison!#0*self.env.action_space.sample()#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    #print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    eshape=elites.shape
                    log.info('eshape[0]:%d,eshape[1]:%d,eshape[2]:%d' % (eshape[0],eshape[1],eshape[2]))
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    #print('cbf_alls',cbf_alls)
                    cbf_alls4=cbf_alls[:,:,0:cbfhorizon-1,:]#cbf_alls[:,:,0:self.plan_hor-1,:]#
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls[:,:,0:cbfhorizon,:]-cbf_initalls4#cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #print('cbfdots_alls', cbfdots_alls)  #
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                                            #dim=1)  # sum over planning horizon#f_G in the paper(1000,1)
                    #cbfdots_violss = torch.sum(cbfdots_alls < acbfs,#the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    #print('cbf_initalls4', cbf_initalls4)  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls,dim=0) < acbfs,  # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbfdots_violss = torch.sum(torch.mean(cbfdots_alls, dim=0) < torch.mean(acbfs,dim=0),# the acbfs is subject to change
                                               dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements####
                    #print('lhse.shape',lhse.shape)
                    #rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices####
                    #print('rhse.shape', rhse.shape)
                    #cbfdots_violss = torch.sum(( lhse< rhse),dim=1) ##### the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('cbfdots_violss',cbfdots_violss)
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 10*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 100*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(),randflag#, tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplanareacheronestd(self, obs):#,state):#,tp,fp,fn,tn,tpc,fpc,fnc,tncsome intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        #embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        randflag=0
        cbfhorizon=self.plan_hor
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    if self.reduce_horizon=='no':
                        act_cbfd_thresh *= 1  # *0.8 by default
                        #act_cbfd_thresh=min(act_cbfd_thresh,1)
                        log.info('alpha keeps %f'%(act_cbfd_thresh))
                    elif self.reduce_horizon=='alpha':
                        act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                        act_cbfd_thresh=min(act_cbfd_thresh,1)
                        log.info('alpha increased to %f'%(act_cbfd_thresh))
                    elif self.reduce_horizon=='horizon':
                        cbfhorizon-=1
                        cbfhorizon=max(1,cbfhorizon)
                        log.info('horizon reduced to %d'%(cbfhorizon))
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('no trajectory candidates satisfy constraints! The BF is doing its job? Picking random actions!')
                        #log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            #tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        randflag=1
                        if self.action_type=='random':
                            return self.env.action_space.sample(),randflag#for fair comparison#
                        elif self.action_type=='zero':
                            return 0*self.env.action_space.sample(),randflag#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#
                        elif self.action_type=='recovery':
                            #do CEM just to max the barrier function value
                            itrrecovery = 0#start from 0
                            while itrrecovery < self.max_iters:#5
                                if itrrecovery == 0:
                                    # Action samples dim (num_candidates, planning_hor, d_act)
                                    if self.mean is None:#right after reset
                                        action_samples = self._sample_actions_random()#1000*5 2d array
                                    else:
                                        num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                                        num_dist = self.popsize - num_random#=0 when random_percent=1
                                        action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)#uniformly random from last iter ation
                                        action_samples_random = self._sample_actions_random(num_random)#completely random within the action limit!
                                        action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
                                else:
                                    # Chop off the numer of elites so we don't use constraint violating trajectories
                                    iter_num_elites = self.num_elites#min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                                    #what if I change this into num_constraint_satisfying+2?
                                    # Sort
                                    #sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                                    sortid = cbfs.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                                    actions_sorted = action_samples[sortid]
                                    elites = actions_sorted[-iter_num_elites:]#get those elite trajectories
                                    # Refitting to Best Trajs
                                    self.mean, self.std = elites.mean(0), elites.std(0)#you get not none self.mean and self.std, so that it would be a good starting point for the next iteration!
                                    #import ipdb#it seems that they are lucky to run into the following case

                                    action_samples = self._sample_actions_normal(self.mean, self.std)
                                    #print('action_samples', action_samples)#it becomes nan!

                                if itrrecovery < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                                    # dimension (num_models, num_candidates, planning_hor, d_latent)
                                    #print('emb.shape',emb.shape)# torch.Size([1, 32])#print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                                    predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                                    num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                                    first_states = predictions[:, :, 0, :].reshape((num_models * num_candidates, d_latent))#the 20000*32 comes out!
                                    #last_states = predictions[:, :, -1, :].reshape((num_models * num_candidates, d_latent))#the 20000*32 comes out!
                                    #all_values = self.value_function.get_value(last_states, already_embedded=True)
                                    all_cbfs=self.cbfdot_function(first_states, already_embedded=True)
                                    #nans = torch.isnan(all_values)#should get it from the cbfd function!
                                    #all_values[nans] = -1e5
                                    nanscbf = torch.isnan(all_cbfs)#should get it from the cbfd function!
                                    all_cbfs[nanscbf] = -1e4
                                    #values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                                    cbfs,indices=torch.min(all_cbfs.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                                    #line 7 in algorithm 1 in the PETS paper!
                                    #values = values.squeeze()
                                    cbfs=cbfs.squeeze()
                                itrrecovery += 1#CEM Evolution method
                                #print('itrrecovery',itrrecovery)
                            # Return the best action
                            action = actions_sorted[-1][0]#the best one
                            return action.detach().cpu().numpy(),randflag
                        #return self.env.action_space.sample()#for fair comparison!#0*self.env.action_space.sample()#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    #print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    eshape=elites.shape
                    log.info('eshape[0]:%d,eshape[1]:%d,eshape[2]:%d' % (eshape[0],eshape[1],eshape[2]))
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    #print('cbf_alls',cbf_alls)
                    cbf_alls4=cbf_alls[:,:,0:cbfhorizon-1,:]#cbf_alls[:,:,0:self.plan_hor-1,:]#
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls[:,:,0:cbfhorizon,:]-cbf_initalls4#cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #print('cbfdots_alls', cbfdots_alls)  #
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                                            #dim=1)  # sum over planning horizon#f_G in the paper(1000,1)
                    #cbfdots_violss = torch.sum(cbfdots_alls < acbfs,#the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    #print('cbf_initalls4', cbf_initalls4)  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls,dim=0) < acbfs,  # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbfstd,cbfmean=torch.std_mean(cbfdots_alls, dim=0)
                    cbfmeanmstd=cbfmean-cbfstd#cbfmean minus 1 std
                    acbfstd,acbfmean=torch.std_mean(acbfs,dim=0)
                    acbfmeanpstd=acbfmean+acbfstd#acbfmean plus 1 std
                    cbfdots_violss = torch.sum(cbfmeanmstd < acbfmeanpstd,# the acbfs is subject to change
                                               dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements####
                    #print('lhse.shape',lhse.shape)
                    #rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices####
                    #print('rhse.shape', rhse.shape)
                    #cbfdots_violss = torch.sum(( lhse< rhse),dim=1) ##### the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('cbfdots_violss',cbfdots_violss)
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 10*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 100*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(),randflag#, tp,fp,fn,tn,tpc,fpc,fnc,tnc


    def actcbfdsquarelatentplanareachernogoaldense(self, obs):#,state):#,tp,fp,fn,tn,tpc,fpc,fnc,tncsome intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        #embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        #print('env.state',state)
        randflag=0
        cbfhorizon=self.plan_hor
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    if self.reduce_horizon=='no':
                        act_cbfd_thresh *= 1  # *0.8 by default
                        #act_cbfd_thresh=min(act_cbfd_thresh,1)
                        log.info('alpha keeps %f'%(act_cbfd_thresh))
                    elif self.reduce_horizon=='alpha':
                        act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                        act_cbfd_thresh=min(act_cbfd_thresh,1)
                        log.info('alpha increased to %f'%(act_cbfd_thresh))
                    elif self.reduce_horizon=='horizon':
                        cbfhorizon-=1
                        cbfhorizon=max(1,cbfhorizon)
                        log.info('horizon reduced to %d'%(cbfhorizon))
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('no trajectory candidates satisfy constraints! The BF is doing its job? Picking random actions!')
                        #log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            #tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        randflag=1
                        if self.action_type=='random':
                            return self.env.action_space.sample(),randflag#for fair comparison#
                        elif self.action_type=='zero':
                            return 0*self.env.action_space.sample(),randflag#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#
                        elif self.action_type=='recovery':
                            #do CEM just to max the barrier function value
                            itrrecovery = 0#start from 0
                            while itrrecovery < self.max_iters:#5
                                if itrrecovery == 0:
                                    # Action samples dim (num_candidates, planning_hor, d_act)
                                    if self.mean is None:#right after reset
                                        action_samples = self._sample_actions_random()#1000*5 2d array
                                    else:
                                        num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                                        num_dist = self.popsize - num_random#=0 when random_percent=1
                                        action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)#uniformly random from last iter ation
                                        action_samples_random = self._sample_actions_random(num_random)#completely random within the action limit!
                                        action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
                                else:
                                    # Chop off the numer of elites so we don't use constraint violating trajectories
                                    iter_num_elites = self.num_elites#min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                                    #what if I change this into num_constraint_satisfying+2?
                                    # Sort
                                    #sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                                    sortid = cbfs.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                                    actions_sorted = action_samples[sortid]
                                    elites = actions_sorted[-iter_num_elites:]#get those elite trajectories
                                    # Refitting to Best Trajs
                                    self.mean, self.std = elites.mean(0), elites.std(0)#you get not none self.mean and self.std, so that it would be a good starting point for the next iteration!
                                    #import ipdb#it seems that they are lucky to run into the following case

                                    action_samples = self._sample_actions_normal(self.mean, self.std)
                                    #print('action_samples', action_samples)#it becomes nan!

                                if itrrecovery < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                                    # dimension (num_models, num_candidates, planning_hor, d_latent)
                                    #print('emb.shape',emb.shape)# torch.Size([1, 32])#print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                                    predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                                    num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                                    first_states = predictions[:, :, 0, :].reshape((num_models * num_candidates, d_latent))#the 20000*32 comes out!
                                    #last_states = predictions[:, :, -1, :].reshape((num_models * num_candidates, d_latent))#the 20000*32 comes out!
                                    #all_values = self.value_function.get_value(last_states, already_embedded=True)
                                    all_cbfs=self.cbfdot_function(first_states, already_embedded=True)
                                    #nans = torch.isnan(all_values)#should get it from the cbfd function!
                                    #all_values[nans] = -1e5
                                    nanscbf = torch.isnan(all_cbfs)#should get it from the cbfd function!
                                    all_cbfs[nanscbf] = -1e4
                                    #values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                                    cbfs,indices=torch.min(all_cbfs.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                                    #line 7 in algorithm 1 in the PETS paper!
                                    #values = values.squeeze()
                                    cbfs=cbfs.squeeze()
                                itrrecovery += 1#CEM Evolution method
                                #print('itrrecovery',itrrecovery)
                            # Return the best action
                            action = actions_sorted[-1][0]#the best one
                            return action.detach().cpu().numpy(),randflag
                        #return self.env.action_space.sample()#for fair comparison#0*self.env.action_space.sample()#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    #print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    eshape=elites.shape
                    log.info('eshape[0]:%d,eshape[1]:%d,eshape[2]:%d' % (eshape[0],eshape[1],eshape[2]))
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                for i in range(planning_hor-1):
                    statesi = predictions[:, :, -1-i-1, :].reshape((num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                    all_valuesi = self.value_function.get_value(statesi, already_embedded=True)#all values from 1000 candidates*20 particles
                    nansi = torch.isnan(all_valuesi)
                    all_valuesi[nansi] = -1e5
                    valuesi = torch.mean(all_valuesi.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                    values+=valuesi#to make it dense
                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    #print('cbf_alls',cbf_alls)
                    cbf_alls4=cbf_alls[:,:,0:cbfhorizon-1,:]#cbf_alls[:,:,0:self.plan_hor-1,:]#
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls[:,:,0:cbfhorizon,:]-cbf_initalls4#cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #print('cbfdots_alls', cbfdots_alls)  #
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                                            #dim=1)  # sum over planning horizon#f_G in the paper(1000,1)
                    #cbfdots_violss = torch.sum(cbfdots_alls < acbfs,#the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    #print('cbf_initalls4', cbf_initalls4)  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls,dim=0) < acbfs,  # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls, dim=0) < torch.mean(acbfs,dim=0),
                                               # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements
                    #print('lhse.shape',lhse.shape)
                    rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices
                    #print('rhse.shape', rhse.shape)
                    cbfdots_violss = torch.sum(( lhse< rhse),dim=1) # the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('cbfdots_violss',cbfdots_violss)
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##

                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5# + goal_states#equation 2 in paper!
                #values = 10*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 100*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(),randflag#, tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplanareacheraveragenogoaldense(self, obs):#,state):#,tp,fp,fn,tn,tpc,fpc,fnc,tncsome intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        #embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        randflag=0
        cbfhorizon=self.plan_hor
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('no trajectory candidates satisfy constraints! The BF is doing its job? Picking random actions!')
                        #log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            #tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        randflag=1
                        if self.action_type=='random':
                            return self.env.action_space.sample(),randflag#for fair comparison#
                        elif self.action_type=='zero':
                            return 0*self.env.action_space.sample(),randflag#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#
                        #return self.env.action_space.sample()#for fair comparison with LS3#0*self.env.action_space.sample()#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    #print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    eshape=elites.shape
                    log.info('eshape[0]:%d,eshape[1]:%d,eshape[2]:%d' % (eshape[0],eshape[1],eshape[2]))
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                for i in range(planning_hor-1):
                    statesi = predictions[:, :, -1-i-1, :].reshape((num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                    all_valuesi = self.value_function.get_value(statesi, already_embedded=True)#all values from 1000 candidates*20 particles
                    nansi = torch.isnan(all_valuesi)
                    all_valuesi[nansi] = -1e5
                    valuesi = torch.mean(all_valuesi.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                    values+=valuesi#to make it dense
                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    #print('cbf_alls',cbf_alls)
                    cbf_alls4=cbf_alls[:,:,0:cbfhorizon-1,:]#cbf_alls[:,:,0:self.plan_hor-1,:]#
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls[:,:,0:cbfhorizon,:]-cbf_initalls4#cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #print('cbfdots_alls', cbfdots_alls)  #
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                                            #dim=1)  # sum over planning horizon#f_G in the paper(1000,1)
                    #cbfdots_violss = torch.sum(cbfdots_alls < acbfs,#the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    #print('cbf_initalls4', cbf_initalls4)  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls,dim=0) < acbfs,  # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbfdots_violss = torch.sum(torch.mean(cbfdots_alls, dim=0) < torch.mean(acbfs,dim=0),# the acbfs is subject to change
                                               dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#this is the right implementation of the average
                    #lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements####
                    #print('lhse.shape',lhse.shape)
                    #rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices####
                    #print('rhse.shape', rhse.shape)
                    #cbfdots_violss = torch.sum(( lhse< rhse),dim=1) ##### the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('cbfdots_violss',cbfdots_violss)
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5# + goal_states#equation 2 in paper!
                #values = 10*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                #values = 100*values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(),randflag#, tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplanaexpensive(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc,obs_relative):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        embexpensive = self.encoder2.encode(obs)  # in latent space now!
        embrepeatexpensive = embexpensive.repeat(self.popsize, self.plan_hor,
                               1)  # emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20expensive = embexpensive.repeat(self.n_particles, self.popsize, 1, 1)  # with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        if self.action_type=='random':
                            return self.env.action_space.sample()#for fair comparison#
                        elif self.action_type=='zero':
                            return 0*self.env.action_space.sample()#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#
                        #return 0*self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    eshape=elites.shape
                    #print('elites.shape',eshape)##print('nan',self.std[0,0])
                    log.info('eshape[0]:%d,eshape[1]:%d,eshape[2]:%d,current state x:%f, current state y:%f' % (eshape[0],eshape[1],eshape[2],state[0],state[1]))
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                predictionsexpensive = self.dynamics_model.predict(embexpensive, action_samples, already_embedded=True)  # (20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                storch=ptu.torchify(state)#state torch
                se=storch+action_samples#se means state estimated#shape(1000,5,2)#se1=stateevolve
                #the square part
                xmove=0#-25#30#
                ymove=0#-45#-40#-35#-33#-30#-25#
                lux = 105#75#50#
                luy = 55#40#
                width = 25#20#
                height = 40# 50#
                walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))]  #
                # I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 0] - walls[0][0][0], se[:, :, 0])
                # Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                # and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1 = torch.concat(
                    (rd1h.reshape(rd1h.shape[0], rd1h.shape[1], 1), rd1v.reshape(rd1v.shape[0], rd1v.shape[1], 1)),
                    dim=2)
                # we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   0 * rd1[:, :, 0], rd1[:, :, 0])  # region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition = (rd2[:, :, 0] > walls[0][1][0]) * (
                            rd2[:, :, 1] <= walls[0][0][1])  # this condition is to see if it is in region 3
                rd3h = torch.where(rd3condition, rd2[:, :, 0] - walls[0][1][0], rd2[:, :, 0])  # h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])  # v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1]) * (
                            rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0 * rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1] - walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) * (rd5[:, :, 0] > walls[0][0][0]) * (
                            rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0 * rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0] - walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) * (rd7[:, :, 1] <= walls[0][1][1]) * (
                            rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0 * rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8s = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)
                rdns = torch.norm(rd8s, dim=2)  # rdn for relative distance norm
                rdnvs = rdns < 15  # rdnv for rdn violator
                rdnvis = torch.sum(rdnvs, dim=1)  # rdn violator indices# print('rdnvi', rdnvi)
                rdnvis = rdnvis.reshape(rdnvis.shape[0], 1)
                rdnvcs = rdns < 10  # rdnv for rdn violator critical
                rdnvcis = torch.sum(rdnvcs, dim=1)  # rdn violator critical indices# print('rdnvci', rdnvci)
                rdnvcis = rdnvcis.reshape(rdnvis.shape[0], 1)# print(rdn.shape)#torch.Size([1000, 5])
                cbfs = rdns ** 2 - 15 ** 2  # 13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                #acbfs = -cbfs * act_cbfd_thresh  # acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                #rdas = torch.concat((rd8s, action_samples),
                                   #dim=2)  # check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network
                #print('action_samples.shape',action_samples.shape)
                #rdas=torch.concat((embrepeat, action_samples),dim=2)

                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    #cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    #cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    cbf_init = self.cbfdot_function(embrepeat20expensive,
                                                    already_embedded=True)  # should have dim (20,1000,1,32) to (20,1000,1,1)
                    # print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictionsexpensive,
                                                    already_embedded=True)  # with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    cbf_alls4=cbf_alls[:,:,0:self.plan_hor-1,:]
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                                            #dim=1)  # sum over planning horizon#f_G in the paper(1000,1)
                    #cbfdots_violss = torch.sum(cbfdots_alls < acbfs,#the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls,dim=0) < acbfs,  # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls, dim=0) < torch.mean(acbfs,dim=0),
                                               # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements
                    #print('lhse.shape',lhse.shape)
                    rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices
                    #print('rhse.shape', rhse.shape)
                    cbfdots_violss = torch.sum(( lhse< rhse),dim=1) # the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                if torch.max(rdnvis) > 0 or torch.max(cbfdots_violss) > 0:  ##
                    rdnvimasks = rdnvis > 0.5
                    cbfdots_violsmasks = cbfdots_violss > 0.5
                    rdnvnotimasks = rdnvis< 0.5
                    cbfdots_notviolsmasks = cbfdots_violss < 0.5
                    tpmasks = rdnvimasks * cbfdots_violsmasks
                    fpmasks = rdnvnotimasks * cbfdots_violsmasks
                    fnmasks = rdnvimasks * cbfdots_notviolsmasks
                    tnmasks = rdnvnotimasks * cbfdots_notviolsmasks
                    tpcounts = torch.sum(tpmasks)
                    fpcounts = torch.sum(fpmasks)
                    fncounts = torch.sum(fnmasks)
                    tncounts = torch.sum(tnmasks)
                    tp += tpcounts
                    fp += fpcounts
                    fn += fncounts
                    tn += tncounts

                    rdnvcicmasks = rdnvcis > 0.5
                    rdnvnotcimasks = rdnvcis < 0.5
                    tpcmasks = rdnvcicmasks * cbfdots_violsmasks
                    fpcmasks = rdnvnotcimasks * cbfdots_violsmasks
                    fncmasks = rdnvcicmasks * cbfdots_notviolsmasks
                    tncmasks = rdnvnotcimasks * cbfdots_notviolsmasks
                    tpccounts = torch.sum(tpcmasks)
                    fpccounts = torch.sum(fpcmasks)
                    fnccounts = torch.sum(fncmasks)
                    tnccounts = torch.sum(tncmasks)
                    tpc += tpccounts
                    fpc += fpccounts
                    fnc += fnccounts
                    tnc += tnccounts
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvis.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvcis.shape[0]
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplanaexpensive2(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc,obs_relative):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        obs_relative = ptu.torchify(obs_relative).reshape(1, *self.d_obs)#just some data processing
        embexpensive = self.encoder2.encode(obs_relative)  # in latent space now!
        embrepeatexpensive = embexpensive.repeat(self.popsize, self.plan_hor,
                               1)  # emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20expensive = embexpensive.repeat(self.n_particles, self.popsize, 1, 1)  # with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        if self.action_type=='random':
                            return self.env.action_space.sample()#for fair comparison#
                        elif self.action_type=='zero':
                            return 0*self.env.action_space.sample()#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#
                        #return 0*self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                predictionsexpensive = self.dynamics_model2.predict(embexpensive, action_samples, already_embedded=True)  # (20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                storch=ptu.torchify(state)#state torch
                se=storch+action_samples#se means state estimated#shape(1000,5,2)#se1=stateevolve
                #the square part
                xmove=0#-25#30#
                ymove=0#-45#-40#-35#-33#-30#-25#
                lux = 50#75#105#
                luy = 55#40#
                width = 25#20#
                height = 40# 50#
                walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))]  #
                # I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 0] - walls[0][0][0], se[:, :, 0])
                # Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                # and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1 = torch.concat(
                    (rd1h.reshape(rd1h.shape[0], rd1h.shape[1], 1), rd1v.reshape(rd1v.shape[0], rd1v.shape[1], 1)),
                    dim=2)
                # we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   0 * rd1[:, :, 0], rd1[:, :, 0])  # region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition = (rd2[:, :, 0] > walls[0][1][0]) * (
                            rd2[:, :, 1] <= walls[0][0][1])  # this condition is to see if it is in region 3
                rd3h = torch.where(rd3condition, rd2[:, :, 0] - walls[0][1][0], rd2[:, :, 0])  # h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])  # v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1]) * (
                            rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0 * rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1] - walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) * (rd5[:, :, 0] > walls[0][0][0]) * (
                            rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0 * rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0] - walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) * (rd7[:, :, 1] <= walls[0][1][1]) * (
                            rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0 * rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8s = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)
                rdns = torch.norm(rd8s, dim=2)  # rdn for relative distance norm
                rdnvs = rdns < 15  # rdnv for rdn violator
                rdnvis = torch.sum(rdnvs, dim=1)  # rdn violator indices# print('rdnvi', rdnvi)
                rdnvis = rdnvis.reshape(rdnvis.shape[0], 1)
                rdnvcs = rdns < 10  # rdnv for rdn violator critical
                rdnvcis = torch.sum(rdnvcs, dim=1)  # rdn violator critical indices# print('rdnvci', rdnvci)
                rdnvcis = rdnvcis.reshape(rdnvis.shape[0], 1)# print(rdn.shape)#torch.Size([1000, 5])
                cbfs = rdns ** 2 - 15 ** 2  # 13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                #acbfs = -cbfs * act_cbfd_thresh  # acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                #rdas = torch.concat((rd8s, action_samples),
                                   #dim=2)  # check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network
                #print('action_samples.shape',action_samples.shape)
                #rdas=torch.concat((embrepeat, action_samples),dim=2)

                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    #cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    #cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    cbf_init = self.cbfdot_function(embrepeat20expensive,
                                                    already_embedded=True)  # should have dim (20,1000,1,32) to (20,1000,1,1)
                    # print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictionsexpensive,
                                                    already_embedded=True)  # with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    cbf_alls4=cbf_alls[:,:,0:self.plan_hor-1,:]
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                                            #dim=1)  # sum over planning horizon#f_G in the paper(1000,1)
                    #cbfdots_violss = torch.sum(cbfdots_alls < acbfs,#the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls,dim=0) < acbfs,  # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls, dim=0) < torch.mean(acbfs,dim=0),
                                               # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements
                    #print('lhse.shape',lhse.shape)
                    rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices
                    #print('rhse.shape', rhse.shape)
                    cbfdots_violss = torch.sum(( lhse< rhse),dim=1) # the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                if torch.max(rdnvis) > 0 or torch.max(cbfdots_violss) > 0:  ##
                    rdnvimasks = rdnvis > 0.5
                    cbfdots_violsmasks = cbfdots_violss > 0.5
                    rdnvnotimasks = rdnvis< 0.5
                    cbfdots_notviolsmasks = cbfdots_violss < 0.5
                    tpmasks = rdnvimasks * cbfdots_violsmasks
                    fpmasks = rdnvnotimasks * cbfdots_violsmasks
                    fnmasks = rdnvimasks * cbfdots_notviolsmasks
                    tnmasks = rdnvnotimasks * cbfdots_notviolsmasks
                    tpcounts = torch.sum(tpmasks)
                    fpcounts = torch.sum(fpmasks)
                    fncounts = torch.sum(fnmasks)
                    tncounts = torch.sum(tnmasks)
                    tp += tpcounts
                    fp += fpcounts
                    fn += fncounts
                    tn += tncounts

                    rdnvcicmasks = rdnvcis > 0.5
                    rdnvnotcimasks = rdnvcis < 0.5
                    tpcmasks = rdnvcicmasks * cbfdots_violsmasks
                    fpcmasks = rdnvnotcimasks * cbfdots_violsmasks
                    fncmasks = rdnvcicmasks * cbfdots_notviolsmasks
                    tncmasks = rdnvnotcimasks * cbfdots_notviolsmasks
                    tpccounts = torch.sum(tpcmasks)
                    fpccounts = torch.sum(fpcmasks)
                    fnccounts = torch.sum(fncmasks)
                    tnccounts = torch.sum(tncmasks)
                    tpc += tpccounts
                    fpc += fpccounts
                    fnc += fnccounts
                    tnc += tnccounts
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvis.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvcis.shape[0]
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplanb(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!#32, also take its last dim now!
        embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        return self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)#now take the last dim
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                storch=ptu.torchify(state)#state torch
                se=storch+action_samples#se means state estimated#shape(1000,5,2)#se1=stateevolve
                #the square part
                xmove=0#-25#30#
                ymove=0#-45#-40#-35#-33#-30#-25#
                lux = 105#50#75#
                luy = 40#55#
                width = 25#20#
                height = 40# 50#
                walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))]  #
                # I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 0] - walls[0][0][0], se[:, :, 0])
                # Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                # and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1 = torch.concat(
                    (rd1h.reshape(rd1h.shape[0], rd1h.shape[1], 1), rd1v.reshape(rd1v.shape[0], rd1v.shape[1], 1)),
                    dim=2)
                # we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   0 * rd1[:, :, 0], rd1[:, :, 0])  # region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition = (rd2[:, :, 0] > walls[0][1][0]) * (
                            rd2[:, :, 1] <= walls[0][0][1])  # this condition is to see if it is in region 3
                rd3h = torch.where(rd3condition, rd2[:, :, 0] - walls[0][1][0], rd2[:, :, 0])  # h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])  # v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1]) * (
                            rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0 * rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1] - walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) * (rd5[:, :, 0] > walls[0][0][0]) * (
                            rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0 * rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0] - walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) * (rd7[:, :, 1] <= walls[0][1][1]) * (
                            rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0 * rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8s = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)
                rdns = torch.norm(rd8s, dim=2)  # rdn for relative distance norm
                rdnvs = rdns < 15  # rdnv for rdn violator
                rdnvis = torch.sum(rdnvs, dim=1)  # rdn violator indices# print('rdnvi', rdnvi)
                rdnvis = rdnvis.reshape(rdnvis.shape[0], 1)
                rdnvcs = rdns < 10  # rdnv for rdn violator critical
                rdnvcis = torch.sum(rdnvcs, dim=1)  # rdn violator critical indices# print('rdnvci', rdnvci)
                rdnvcis = rdnvcis.reshape(rdnvis.shape[0], 1)# print(rdn.shape)#torch.Size([1000, 5])
                cbfs = rdns ** 2 - 15 ** 2  # 13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                #acbfs = -cbfs * act_cbfd_thresh  # acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                #rdas = torch.concat((rd8s, action_samples),
                                   #dim=2)  # check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network
                #print('action_samples.shape',action_samples.shape)
                #rdas=torch.concat((embrepeat, action_samples),dim=2)

                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbf_init13 =embrepeat20[:,:,:,-1]#self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    cbf_init=torch.pow(cbf_init13, 3)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1])#torch.Size([20, 1000, 1, 1])
                    cbf_alls13 = predictions[:,:,:,-1]#self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    cbf_alls=torch.pow(cbf_alls13, 3)
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5])#torch.Size([20, 1000, 5, 1])
                    cbf_alls4=cbf_alls[:,:,0:self.plan_hor-1]#cbf_alls[:,:,0:self.plan_hor-1,:]#
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=2)
                    #print('cbf_initalls4.shape', cbf_initalls4.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                                            #dim=1)  # sum over planning horizon#f_G in the paper(1000,1)
                    #cbfdots_violss = torch.sum(cbfdots_alls < acbfs,#the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls,dim=0) < acbfs,  # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #cbfdots_violss = torch.sum(torch.mean(cbfdots_alls, dim=0) < torch.mean(acbfs,dim=0),
                                               # the acbfs is subject to change
                                               #dim=1)  # those that violate the constraints#1000 0,1,2,3,4,5s#
                    lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements
                    #print('lhse.shape',lhse.shape)
                    rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices
                    #print('rhse.shape', rhse.shape)
                    cbfdots_violss = torch.sum(( lhse< rhse),dim=1) # the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                if torch.max(rdnvis) > 0 or torch.max(cbfdots_violss) > 0:  ##
                    rdnvimasks = rdnvis > 0.5
                    cbfdots_violsmasks = cbfdots_violss > 0.5
                    rdnvnotimasks = rdnvis< 0.5
                    cbfdots_notviolsmasks = cbfdots_violss < 0.5
                    tpmasks = rdnvimasks * cbfdots_violsmasks
                    fpmasks = rdnvnotimasks * cbfdots_violsmasks
                    fnmasks = rdnvimasks * cbfdots_notviolsmasks
                    tnmasks = rdnvnotimasks * cbfdots_notviolsmasks
                    tpcounts = torch.sum(tpmasks)
                    fpcounts = torch.sum(fpmasks)
                    fncounts = torch.sum(fnmasks)
                    tncounts = torch.sum(tnmasks)
                    tp += tpcounts
                    fp += fpcounts
                    fn += fncounts
                    tn += tncounts

                    rdnvcicmasks = rdnvcis > 0.5
                    rdnvnotcimasks = rdnvcis < 0.5
                    tpcmasks = rdnvcicmasks * cbfdots_violsmasks
                    fpcmasks = rdnvnotcimasks * cbfdots_violsmasks
                    fncmasks = rdnvcicmasks * cbfdots_notviolsmasks
                    tncmasks = rdnvnotcimasks * cbfdots_notviolsmasks
                    tpccounts = torch.sum(tpcmasks)
                    fpccounts = torch.sum(fpcmasks)
                    fnccounts = torch.sum(fncmasks)
                    tnccounts = torch.sum(tncmasks)
                    tpc += tpccounts
                    fpc += fpccounts
                    fnc += fnccounts
                    tnc += tnccounts
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvis.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvcis.shape[0]
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplananogoal(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        if self.action_type=='random':
                            return self.env.action_space.sample()#for fair comparison#
                        elif self.action_type=='zero':
                            return 0*self.env.action_space.sample()#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#
                        #return 0*self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                storch=ptu.torchify(state)#state torch
                se=storch+action_samples#se means state estimated#shape(1000,5,2)#se1=stateevolve
                #the square part
                xmove=0#-25#30#
                ymove=0#-45#-40#-35#-33#-30#-25#
                lux = 75#105#50#
                luy = 55#40#
                width = 25#20#
                height = 40# 50#
                walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))]  #

                # I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 0] - walls[0][0][0], se[:, :, 0])
                # Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                # and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1 = torch.concat(
                    (rd1h.reshape(rd1h.shape[0], rd1h.shape[1], 1), rd1v.reshape(rd1v.shape[0], rd1v.shape[1], 1)),
                    dim=2)
                # we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   0 * rd1[:, :, 0], rd1[:, :, 0])  # region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition = (rd2[:, :, 0] > walls[0][1][0]) * (
                            rd2[:, :, 1] <= walls[0][0][1])  # this condition is to see if it is in region 3
                rd3h = torch.where(rd3condition, rd2[:, :, 0] - walls[0][1][0], rd2[:, :, 0])  # h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])  # v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1]) * (
                            rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0 * rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1] - walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) * (rd5[:, :, 0] > walls[0][0][0]) * (
                            rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0 * rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0] - walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) * (rd7[:, :, 1] <= walls[0][1][1]) * (
                            rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0 * rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8s = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)
                rdns = torch.norm(rd8s, dim=2)  # rdn for relative distance norm
                rdnvs = rdns < 15  # rdnv for rdn violator
                rdnvis = torch.sum(rdnvs, dim=1)  # rdn violator indices# print('rdnvi', rdnvi)
                rdnvis = rdnvis.reshape(rdnvis.shape[0], 1)
                rdnvcs = rdns < 10  # rdnv for rdn violator critical
                rdnvcis = torch.sum(rdnvcs, dim=1)  # rdn violator critical indices# print('rdnvci', rdnvci)
                rdnvcis = rdnvcis.reshape(rdnvis.shape[0], 1)# print(rdn.shape)#torch.Size([1000, 5])

                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    #print('cbf_alls',cbf_alls)
                    cbf_alls4=cbf_alls[:,:,0:self.plan_hor-1,:]
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #print('cbfdots_alls', cbfdots_alls)  #
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    #print('cbf_initalls4', cbf_initalls4)  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements
                    #print('lhse.shape',lhse.shape)
                    rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices
                    #print('rhse.shape', rhse.shape)
                    cbfdots_violss = torch.sum(( lhse< rhse),dim=1) # the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('cbfdots_violss',cbfdots_violss)
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                if torch.max(rdnvis) > 0 or torch.max(cbfdots_violss) > 0:  ##
                    rdnvimasks = rdnvis > 0.5
                    cbfdots_violsmasks = cbfdots_violss > 0.5
                    rdnvnotimasks = rdnvis< 0.5
                    cbfdots_notviolsmasks = cbfdots_violss < 0.5
                    tpmasks = rdnvimasks * cbfdots_violsmasks
                    fpmasks = rdnvnotimasks * cbfdots_violsmasks
                    fnmasks = rdnvimasks * cbfdots_notviolsmasks
                    tnmasks = rdnvnotimasks * cbfdots_notviolsmasks
                    tpcounts = torch.sum(tpmasks)
                    fpcounts = torch.sum(fpmasks)
                    fncounts = torch.sum(fnmasks)
                    tncounts = torch.sum(tnmasks)
                    tp += tpcounts
                    fp += fpcounts
                    fn += fncounts
                    tn += tncounts

                    rdnvcicmasks = rdnvcis > 0.5
                    rdnvnotcimasks = rdnvcis < 0.5
                    tpcmasks = rdnvcicmasks * cbfdots_violsmasks
                    fpcmasks = rdnvnotcimasks * cbfdots_violsmasks
                    fncmasks = rdnvcicmasks * cbfdots_notviolsmasks
                    tncmasks = rdnvnotcimasks * cbfdots_notviolsmasks
                    tpccounts = torch.sum(tpcmasks)
                    fpccounts = torch.sum(fpcmasks)
                    fnccounts = torch.sum(fncmasks)
                    tnccounts = torch.sum(tncmasks)
                    tpc += tpccounts
                    fpc += fpccounts
                    fnc += fnccounts
                    tnc += tnccounts
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvis.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvcis.shape[0]
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                #goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 #+ goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc

    def actcbfdsquarelatentplananogoaldense(self, obs,state,tp,fp,fn,tn,tpc,fpc,fnc,tnc):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.
        Arguments:obs: The current observation. Cannot be a batch
        Returns: An action (and possibly the predicted cost)
        """
        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!
        embrepeat=emb.repeat(self.popsize,self.plan_hor,1)#emb.repeat(1000,5,1), with new shape (1000,5,32)#1000 and 5 should subject to change#print('embrepeat.shape',embrepeat.shape)
        embrepeat20 = emb.repeat(self.n_particles, self.popsize, 1, 1)  #with new shape (20,1000,1,32)#
        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.cbfd_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation#print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.cbfd_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                            tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                        if self.action_type=='random':
                            return self.env.action_space.sample()#for fair comparison#
                        elif self.action_type=='zero':
                            return 0*self.env.action_space.sample()#,tp,fp,fn,tn,tpc,fpc,fnc,tnc#
                        #return 0*self.env.action_space.sample(),tp,fp,fn,tn,tpc,fpc,fnc,tnc#really random action!
                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue
                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)# print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!#print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)##print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)#0.8 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)#print('action_samples', action_samples)#it becomes nan!
            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)#(20,1000,5,32)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials
                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                for i in range(planning_hor-1):
                    statesi = predictions[:, :, -1-i-1, :].reshape((num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                    all_valuesi = self.value_function.get_value(statesi, already_embedded=True)#all values from 1000 candidates*20 particles
                    nansi = torch.isnan(all_valuesi)
                    all_valuesi[nansi] = -1e5
                    valuesi = torch.mean(all_valuesi.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20
                    values+=valuesi#to make it dense
                storch=ptu.torchify(state)#state torch
                se=storch+action_samples#se means state estimated#shape(1000,5,2)#se1=stateevolve
                #the square part
                xmove=0#-25#30#
                ymove=0#-45#-40#-35#-33#-30#-25#
                lux = 75#105#50#
                luy = 55#40#
                width = 25#20#
                height = 40# 50#
                walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))]  #

                # I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 0] - walls[0][0][0], se[:, :, 0])
                # Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),
                                   se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                # and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1 = torch.concat(
                    (rd1h.reshape(rd1h.shape[0], rd1h.shape[1], 1), rd1v.reshape(rd1v.shape[0], rd1v.shape[1], 1)),
                    dim=2)
                # we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   0 * rd1[:, :, 0], rd1[:, :, 0])  # region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0]) * (rd1[:, :, 0] <= walls[0][1][0]) * (
                            rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition = (rd2[:, :, 0] > walls[0][1][0]) * (
                            rd2[:, :, 1] <= walls[0][0][1])  # this condition is to see if it is in region 3
                rd3h = torch.where(rd3condition, rd2[:, :, 0] - walls[0][1][0], rd2[:, :, 0])  # h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])  # v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1]) * (
                            rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0 * rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1] - walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) * (rd5[:, :, 0] > walls[0][0][0]) * (
                            rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0 * rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0] - walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) * (rd7[:, :, 1] <= walls[0][1][1]) * (
                            rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0 * rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8s = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)  # dim: (1000,5,2)
                rdns = torch.norm(rd8s, dim=2)  # rdn for relative distance norm
                rdnvs = rdns < 5#15  # rdnv for rdn violator
                rdnvis = torch.sum(rdnvs, dim=1)  # rdn violator indices# print('rdnvi', rdnvi)
                rdnvis = rdnvis.reshape(rdnvis.shape[0], 1)
                rdnvcs = rdns < 1e-8#10  # rdnv for rdn violator critical
                rdnvcis = torch.sum(rdnvcs, dim=1)  # rdn violator critical indices# print('rdnvci', rdnvci)
                rdnvcis = rdnvcis.reshape(rdnvis.shape[0], 1)# print(rdn.shape)#torch.Size([1000, 5])

                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    #some sudo code:
                    #1.cbf_alls=self.cbf_function(prediction)
                    #2.Dislocation Subtraction
                    #3.some sum (torch.sum)#2 versions: (20,1000,5) and (1000,5)
                    #4.
                    #cbfdots_alls = self.cbfdot_function(rdas,
                                                        #already_embedded=True)  # all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    cbf_init = self.cbfdot_function(embrepeat20, already_embedded=True)#should have dim (20,1000,1,32) to (20,1000,1,1)
                    #print('cbf_init.shape',cbf_init.shape)#torch.Size([20, 1000, 1, 1])
                    cbf_alls = self.cbfdot_function(predictions,already_embedded=True) #with the reformulated cbfd estimator
                    #print('cbf_alls.shape',cbf_alls.shape)#torch.Size([20, 1000, 5, 1])
                    #print('cbf_alls',cbf_alls)
                    cbf_alls4=cbf_alls[:,:,0:self.plan_hor-1,:]
                    #print('cbf_alls4.shape', cbf_alls4.shape)#torch.Size([20, 1000, 4, 1])
                    cbf_initalls4=torch.cat((cbf_init,cbf_alls4),dim=-2)
                    #print('cbf_initalls.shape', cbf_initalls.shape)#torch.Size([20, 1000, 5, 1])
                    cbfdots_alls=cbf_alls-cbf_initalls4#the mean is also subject to change
                    cbfdots_alls = cbfdots_alls.reshape(cbfdots_alls.shape[0], cbfdots_alls.shape[1],cbfdots_alls.shape[2])  #
                    #print('cbfdots_alls.shape',cbfdots_alls.shape)#torch.Size([20, 1000, 5])
                    #print('cbfdots_alls', cbfdots_alls)  #
                    #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh,
                    cbf_initalls4 = cbf_initalls4.reshape(cbf_initalls4.shape[0], cbf_initalls4.shape[1],
                                                        cbf_initalls4.shape[2])  #
                    #print('cbf_initalls4', cbf_initalls4)  #
                    acbfs = -act_cbfd_thresh * cbf_initalls4  #
                    #print('acbfs.shape',acbfs.shape)#torch.Size([20, 1000, 5])right#torch.Size([20, 1000, 5, 1])wrong#
                    lhse,lhsi=torch.min(cbfdots_alls, dim=0)#lhse means left hand side elements
                    #print('lhse.shape',lhse.shape)
                    rhse,rhsi=torch.max(acbfs, dim=0)#rhsi means right hand side indices
                    #print('rhse.shape', rhse.shape)
                    cbfdots_violss = torch.sum(( lhse< rhse),dim=1) # the acbfs is subject to change # those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('cbfdots_violss',cbfdots_violss)
                    cbfdots_violss = cbfdots_violss.reshape(cbfdots_violss.shape[0],1)  # the threshold now should be predictions dependent
                else:#if ignoring the cbf dot constraints#in new setting I need Dislocation Subtraction
                    cbfdots_violss = torch.zeros((num_candidates, 1),
                                                 device=ptu.TORCH_DEVICE)  # no constraint violators!
                #self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                #if torch.max(rdnvis)>0 or torch.max(cbfdots_violss)>0 or torch.max(rdnvic)>0 or torch.max(cbfdots_violsc)>0:##
                if torch.max(rdnvis) > 0 or torch.max(cbfdots_violss) > 0:  ##
                    rdnvimasks = rdnvis > 0.5
                    cbfdots_violsmasks = cbfdots_violss > 0.5
                    rdnvnotimasks = rdnvis< 0.5
                    cbfdots_notviolsmasks = cbfdots_violss < 0.5
                    tpmasks = rdnvimasks * cbfdots_violsmasks
                    fpmasks = rdnvnotimasks * cbfdots_violsmasks
                    fnmasks = rdnvimasks * cbfdots_notviolsmasks
                    tnmasks = rdnvnotimasks * cbfdots_notviolsmasks
                    tpcounts = torch.sum(tpmasks)
                    fpcounts = torch.sum(fpmasks)
                    fncounts = torch.sum(fnmasks)
                    tncounts = torch.sum(tnmasks)
                    tp += tpcounts
                    fp += fpcounts
                    fn += fncounts
                    tn += tncounts

                    rdnvcicmasks = rdnvcis > 0.5
                    rdnvnotcimasks = rdnvcis < 0.5
                    tpcmasks = rdnvcicmasks * cbfdots_violsmasks
                    fpcmasks = rdnvnotcimasks * cbfdots_violsmasks
                    fncmasks = rdnvcicmasks * cbfdots_notviolsmasks
                    tncmasks = rdnvnotcimasks * cbfdots_notviolsmasks
                    tpccounts = torch.sum(tpcmasks)
                    fpccounts = torch.sum(fpcmasks)
                    fnccounts = torch.sum(fncmasks)
                    tnccounts = torch.sum(tncmasks)
                    tpc += tpccounts
                    fpc += fpccounts
                    fnc += fnccounts
                    tnc += tnccounts
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,itr:%d,current state x:%f, current state y:%f' % (
                    tp, fp, fn, tn, tpc, fpc, fnc, tnc,itr,state[0],state[1]))
                else:
                    tp = tp
                    fp = fp
                    fn = fn
                    tn = tn+rdnvis.shape[0]
                    tpc = tpc
                    fpc = fpc
                    fnc = fnc
                    tnc = tnc + rdnvcis.shape[0]
                #cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)  # no constraint violators!#for testing!
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                #goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                #goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                values = values + (constraint_viols +cbfdots_violss+safe_set_viols) * -1e5 #+ goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5
            itr += 1#CEM Evolution method
        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy(), tp,fp,fn,tn,tpc,fpc,fnc,tnc


    def reset(self):#where it is used?
        # It's important to call this after each episode
        self.mean, self.std = None, None

    def _sample_actions_random(self, n=None):#something totally random!
        if n is None:
            n = self.popsize#1000
        rand = torch.rand((n, self.plan_hor, self.d_act))#(1000,5,2)
        scaled = rand * (self.ac_ub - self.ac_lb)
        action_samples = scaled + self.ac_lb#something random between ac_lb and ac_ub
        return action_samples.to(ptu.TORCH_DEVICE)#size of (1000,5,2)

    def _sample_actions_normal(self, mean, std, n=None):#sample from a normal distribution with mean=mean and std=std
        if n is None:
            n = self.popsize

        smp = torch.empty(n, self.plan_hor, self.d_act).normal_(
            mean=0, std=1).to(ptu.TORCH_DEVICE)
        mean = mean.unsqueeze(0).repeat(n, 1, 1).to(ptu.TORCH_DEVICE)
        std = std.unsqueeze(0).repeat(n, 1, 1).to(ptu.TORCH_DEVICE)

        # Sample new actions
        action_samples = smp * std + mean
        # TODO: Assuming action space is symmetric, true for maze and shelf for now
        action_samples = torch.clamp(
            action_samples,
            min=self.env.action_space.low[0],
            max=self.env.action_space.high[0])

        return action_samples
