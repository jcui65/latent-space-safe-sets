from abc import ABC

from latentsafesets.envs import simple_point_bot as spb

import numpy as np

from scipy import linalg as la
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import BFGS

def mympc(A,B,Q,R,P,N,umin,umax,xmin,xmax,x,xdestin,AXE=np.zeros((1,2)),BXE=1e-6*np.ones(1)):
    x0=x-xdestin#put it inside!
    SX=np.eye(A.shape[0])
    for i in range(N):
        AIP1=np.linalg.matrix_power(A,i+1)#AIP1 means A i plus 1
        SX=np.concatenate((SX,AIP1),axis=0)#calculate the SX matrix in slides 5/6
    SU=np.zeros((B.shape[0]*N,B.shape[1]*N))#B#SUIFULL=[zeros(size(B,1),size(B,2)*N);SUI]
    for i in range(N):
        AIB=np.matmul(np.linalg.matrix_power(A,i),B)
        SUI=np.kron(np.diag(np.ones(N-i),-i),AIB)
        SU=SU+SUI#calculate the SU matrix in slides 5/6%BH = blkdiag(kron(eye(N),Q), P, kron(eye(N),R))
    SU=np.concatenate((np.zeros((B.shape[0],B.shape[1]*N)),SU),axis=0)
    QB=la.block_diag(np.kron(np.eye(N),Q), P)#QB for Q block_diagonal
    RB=la.block_diag(np.kron(np.eye(N),R))#RB for R block_diagonal
    SUT=np.transpose(SU)
    H=np.matmul(np.matmul(SUT,QB),SU)+RB#H is in slides5/6
    f=1*np.matmul(np.matmul(np.matmul(SUT,QB),SX),x0)#2.*SU'*QB*SX*x0#f is in slides5/6#page 28 of slides 6!
    AU=np.array([[1,0],[0,1],[-1,0],[0,-1]])#np.array([[1],[-1]])
    AUD=np.kron(np.eye(N),AU)#D for diagonalization%calculate the AU matrix in slides 6
    AX=np.array([[1,0],[0,1],[-1,0],[0,-1]])#4*2
    AX=np.concatenate((AX,AXE),axis=0)
    AXB=np.matmul(AX,B)
    AXD=np.zeros((AXB.shape[0]*N,AXB.shape[1]*N))#calculate the Ax matrix in slides 6
    for i in range(N):
        AI = np.linalg.matrix_power(A, i)
        AXAB=np.matmul(np.matmul(AX,AI),B)
        AXDI=np.kron(np.diag(np.ones(N-i),-i),AXAB)
        AXD=AXD+AXDI#the last AF is AX
    AXD=np.concatenate((np.zeros((AXB.shape[0]*1,AXB.shape[1]*N)),AXD),axis=0)
    G0=np.concatenate((AUD,AXD),axis=0)#THE G0 matrix in slides 6
    E01=np.zeros((AU.shape[0]*N,A.shape[1]))#E01 for E0 part 1
    AXDREAL=np.kron(np.eye(N+1),AX)#
    E02=-np.matmul(AXDREAL,SX)#E02 for E0 part 2
    E0=np.concatenate((E01,E02),axis=0)#THE E0 matrix in slides 6
    BU=np.array([umax[0],umax[1],-umin[0],-umin[1]])#([umax,-umin])#THE BU matrix in slides 6
    BX=np.array([xmax[0]-xdestin[0,0],xmax[1]-xdestin[1,0],-(xmin[0]-xdestin[0,0]),-xmin[1]+xdestin[1,0]])
    BX=np.concatenate((BX,BXE),axis=0)#THE BX matrix in slides 6
    W0U=np.kron(np.ones((N)),BU)#The first part of W0 matrix
    W0X=np.kron(np.ones((N+1)),BX)#the 2nd part of W0 matrix
    W0=np.concatenate((W0U,W0X),axis=0)#now it is just a row#mind the dim!#THE W0 matrix in slides 6
    #W0=W0.reshape((W0.shape[0],1))#
    W0E0x0=W0+np.matmul(E0,x0).reshape((W0.shape[0],))#it is W0+E0x0 term in slides 6 page 28
    lb=-np.inf*np.ones_like(W0E0x0)
    linear_constraint = LinearConstraint(G0, lb, W0E0x0)
    objective= lambda u: 0.5*np.matmul(np.matmul(np.transpose(u),H),u)+np.matmul(np.transpose(f),u)
    #(x[0] - 1) ** 2 + (x[1] - 2.5) ** 2#u = quadprog(H,f,G0,W0E0x0)#use quadprog to find the sequence of u
    u0=np.ones(N*B.shape[1])##np.ones((N,1))#just an initial value
    res = minimize(objective, u0,constraints=linear_constraint)
    resx=res.x.reshape((N,B.shape[1]))
    return resx#the u is like the x in the quadprog function in matlab documentation
class AbstractTeacher(ABC):

    def __init__(self, env, noisy=False, on_policy=True, horizon=None):
        self.env = env
        self.noisy = noisy
        self.on_policy = on_policy

        self.ac_high = env.action_space.high
        self.ac_low = env.action_space.low
        self.noise_std = (self.ac_high - self.ac_low) / 5
        self.random_start = False
        if horizon is None:
            self.horizon = env.horizon
        else:
            self.horizon = horizon

    def generate_demonstrations(self, num_demos, store_noisy=True, noise_param=None):
        demonstrations = []
        for i in range(num_demos):
            demo = self.generate_trajectory(noise_param, store_noisy=store_noisy)
            reward = sum([frame['reward'] for frame in demo])
            print('Trajectory %d, Reward %d' % (i, reward))
            demonstrations.append(demo)
        return demonstrations

    def generate_trajectory(self, noise_param=None, store_noisy=True):
        """
        The teacher initially tries to go northeast before going to the origin
        """
        self.reset()
        transitions = []#AN EMPTY LIST
        obs = self.env.reset(random_start=self.random_start)#obs is a 3 channel image!
        #around line 85 in simple_point_bot.py#random_start is false by default
        # state = np.zeros((0, 0))
        state = None
        done = False
        for i in range(self.horizon):
            if state is None:
                action = self.env.action_space.sample().astype(np.float64)#sample between -3 and 3
            else:#I think the control is usually either -3 or +3
                action = self._expert_control(state, i).astype(np.float64)
            if self.noisy:
                action_input = np.random.normal(action, self.noise_std)
                action_input = np.clip(action_input, self.ac_low, self.ac_high)
            else:
                action_input = action

            if store_noisy:
                action = action_input#if it not noisy, then it is just the same
            #import ipdb; ipdb.set_trace()
            next_obs, reward, done, info = self.env.step(action_input)#about 63 in simple_point_bot.py
            transition = {'obs': obs, 'action': tuple(action), 'reward': float(reward),
                          'next_obs': next_obs, 'done': int(done),#this is a dictionary
                          'constraint': int(info['constraint']), 'safe_set': 0,
                          'on_policy': int(self.on_policy)}#add key and value into it!
            # print({k: v.dtype for k, v in transition.items() if 'obs' in k})
            transitions.append(transition)#a list of dictionaries!
            state = info['next_state']
            obs = next_obs

            if done:#it is just a time count rather than a sign of success or not!
                break

        transitions[-1]['done'] = 1

        rtg = 0#reward to goal?
        ss = 0
        for frame in reversed(transitions):
            if frame['reward'] >= 0:
                ss = 1
            #along the way of the trajectroy, the trajectory is safe
            frame['safe_set'] = ss#is this dynamic programming?
            frame['rtg'] = rtg#the reward to goal at each frame!#I think this is good
            #add a key value pair to the trajectory(key='rtg', value=rtg
            rtg = rtg + frame['reward']

        # assert done, "Did not reach the goal set on task completion."
        # V = self.env.values()
        # for i, t in enumerate(transitions):
        #     t['values'] = V[i]
        return transitions#100 transitions, one whole trajectory

    def generate_trajectorysafety(self, noise_param=None, store_noisy=True):
        """
        The teacher initially tries to go northeast before going to the origin
        """
        self.reset()
        transitions = []#AN EMPTY LIST
        obs = self.env.reset(random_start=self.random_start)#obs is a 3 channel image!
        #around line 85 in simple_point_bot.py#random_start is false by default
        # state = np.zeros((0, 0))
        state = None
        done = False
        for i in range(self.horizon):
            if state is None:
                action = self.env.action_space.sample().astype(np.float64)#sample between -3 and 3
            else:#I think the control is usually either -3 or +3
                action = self._expert_control(state, i).astype(np.float64)#i is here
            if self.noisy:
                action_input = np.random.normal(action, self.noise_std)
                action_input = np.clip(action_input, self.ac_low, self.ac_high)
            else:
                action_input = action

            if store_noisy:
                action = action_input#if it not noisy, then it is just the same
            #import ipdb; ipdb.set_trace()
            next_obs, reward, done, info = self.env.stepsafety(action_input)#63 in simple_point_bot.py
            transition = {'obs': obs, 'action': tuple(action), 'reward': float(reward),
                          'next_obs': next_obs, 'done': int(done),#this is a dictionary
                          'constraint': int(info['constraint']), 'safe_set': 0,
                          'on_policy': int(self.on_policy),
                          'rdo': info['rdo'].tolist(),
                          'rdn': info['rdn'].tolist(),
                          'hvo': info['hvo'],
                          'hvn': info['hvn'],
                          'hvd': info['hvd'],
                          'state':info['state'].tolist(),
                          'next_state':info['next_state'].tolist()
                          }#add key and value into it!
            # print({k: v.dtype for k, v in transition.items() if 'obs' in k})
            transitions.append(transition)#a list of dictionaries!
            state = info['next_state']
            obs = next_obs

            if done:#it is just a time count rather than a sign of success or not!
                break

        transitions[-1]['done'] = 1

        rtg = 0#reward to goal?
        ss = 0
        for frame in reversed(transitions):
            if frame['reward'] >= 0:
                ss = 1
            #along the way of the trajectroy, the trajectory is safe
            frame['safe_set'] = ss#is this dynamic programming?
            frame['rtg'] = rtg#the reward to goal at each frame!#I think this is good
            #add a key value pair to the trajectory(key='rtg', value=rtg
            rtg = rtg + frame['reward']

        # assert done, "Did not reach the goal set on task completion."
        # V = self.env.values()
        # for i, t in enumerate(transitions):
        #     t['values'] = V[i]
        return transitions

    def _expert_control(self, state, i):
        raise NotImplementedError("Override in subclass")

    def reset(self):
        pass


class SimplePointBotTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):#starting from (30,75)
        super().__init__(env, noisy)
        self.goal = (150, 75)

    def _expert_control(self, s, t):
        if t < 20:#the max speed is 3m/s, thus usually it will take 20 seconds to go from 75 to 15
            goal = np.array((30, 15))
        elif t < 60:#the max speed is 3m/s, thus it will take about 40 seconds to go from 30 to 150
            goal = np.array((150, 15))
        else:#thus the reward is usually around -80
            goal = self.goal

        act = np.subtract(goal, s)
        act = np.clip(act, -3, 3)
        return act

class SimplePointBotTeacherMPC(AbstractTeacher):
    def __init__(self, env, noisy=False):#starting from (30,75)
        super().__init__(env, noisy)
        self.goal = (150, 75)

    def _expert_control(self, s, t):
        if t < 20:#the max speed is 3m/s, thus usually it will take 20 seconds to go from 75 to 15
            goal = np.array((30, 15))
        elif t < 60:#the max speed is 3m/s, thus it will take about 40 seconds to go from 30 to 150
            goal = np.array((150, 15))
        else:#thus the reward is usually around -80
            goal = self.goal
            goal=np.asarray(goal)
        #act = np.subtract(goal, s)
        A = np.array([[1, 0], [0, 1]])  # [EV1,EV2]=linalg.eigvals(A)#ACTUALLY NOT USED
        B = np.array([[1, 0], [0, 1]])  # ([[0.2173],[0.0573]])#what should the dim of B be?
        QV1 = 100#1  # ;#500;#change this in part 8#QV for Q value
        QV2 = 100#1  # 500;#;#change this in part 8#
        Q = np.diag([QV1, QV2])  # QV.*eye(2);%100*eye(2);#
        RV1 = 1;RV2 = 1
        R = np.diag([RV1, RV2])  # R
        N = 5  # 1;#2;#25#10#;#prediction horizon, change this in part 6,7,8#
        #T = 1  # 0.1#sample timeN2 = 25  # 10#5#2#it is the plotting horizon, 5s=50steps
        PRIC = la.solve_discrete_are(A, B, Q, R)  # P_ric
        width=180#bignumber = 100000000  # as if it is infinity
        height=150
        xmin = [-0, -0]  # [-bignumber,-1]#0]#0#-100000#-inf
        xmax = [+width, +height]  # 10000000000#inf
        um=spb.MAX_FORCE
        umin = [-um, -um]  # -5#-50#-100#
        umax = [um, um]  # 5#50#100#
        x0 = s.reshape((s.shape[0],1))#np.array([[6], [60]])  # [0;10];%[0.10;0.10]#x0 is the initial condition, will be changed later#
        XK = x0  #XKA = XK  # [XK]#XKA is XK arrayUKA = np.zeros((B.shape[1], 1))  # UKA is UK array
        #print(goal.shape)
        #print()
        xdestin = goal.reshape((goal.shape[0],1))#np.array([[2], [20]])  #
        #for i in range(int(N2 * (1 / T))):  # u0=mympc(A,B,Q,R,PRIC,N,umin,umax,xmin,xmax,XK)%use P_ric
        u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, XK, xdestin)  # use P_lyap
        # u0=mympc(A,B,Q,R,P,N,umin,umax,xmin,xmax,XK)%use P
        UK = u0[0]  # FKI*XK%pick the first one from u0 to implement
        #UK = UK.reshape((UK.shape[0], 1))#UKA = np.concatenate((UKA, UK), axis=1)  # UKA.append(UK)#np.concatenate(2,UKA,UK)#store the values
        #XKP1 = np.matmul(A, XK) + np.matmul(B,UK)  # B*UK#+0.0*sqrt(10)*randn(2,1)#waiting to add the disturbance term!
        #XKA = np.concatenate((XKA, XKP1), axis=1)  # XKA.append(XKP1)#store the values#XK = XKP1  # XKP1 means x_k+1
        act=UK
        act = np.clip(act, -um, um)
        return act#pay attention to its dimension

class SimplePointBotTeacherMPCCBFD(AbstractTeacher):
    def __init__(self, env, noisy=False):#starting from (30,75)
        super().__init__(env, noisy)
        self.goal = (150, 75)

    def _expert_control(self, s, t):
        if t < 15:#the max speed is 3m/s, thus usually it will take 20 seconds to go from 75 to 15
            goal = np.array((60, 75))
        elif t < 40:#the max speed is 3m/s, thus it will take about 40 seconds to go from 30 to 150
            goal = np.array((100, 40))
        elif t < 55:#the max speed is 3m/s, thus it will take about 40 seconds to go from 30 to 150
            goal = np.array((115, 75))
        else:#thus the reward is usually around -80
            goal = np.asarray(self.goal)#notice the data type and shape
        #act = np.subtract(goal, s)
        A = np.array([[1, 0], [0, 1]])  # [EV1,EV2]=linalg.eigvals(A)#ACTUALLY NOT USED
        B = np.array([[1, 0], [0, 1]])  # ([[0.2173],[0.0573]])#what should the dim of B be?
        QV1 = 100#1  # ;#500;#change this in part 8#QV for Q value
        QV2 = 100#1  # 500;#;#change this in part 8#
        Q = np.diag([QV1, QV2])  # QV.*eye(2);%100*eye(2);#
        RV1 = 1;RV2 = 1
        R = np.diag([RV1, RV2])  # R
        N = 5  # 1;#2;#25#10#;#prediction horizon, change this in part 6,7,8#
        PRIC = la.solve_discrete_are(A, B, Q, R)  # P_ric
        width = 180  # bignumber = 100000000  # as if it is infinity
        height = 150
        xmin = [-0, -0]  # [-bignumber,-1]#0]#0#-100000#-inf
        xmax = [+width, +height]  # 10000000000#inf
        um=spb.MAX_FORCE
        umin = [-um, -um]  # -5#-50#-100#
        umax = [um, um]  # 5#50#100#
        x0 = s.reshape((s.shape[0],1))#np.array([[6], [60]])  # [0;10];%[0.10;0.10]#x0 is the initial condition, will be changed later#
        xdestin = goal.reshape((goal.shape[0],1))#np.array([[2], [20]])  #
        selfwall_coords = [((75, 55), (100, 95))]  # the position and dimension of the wall
        leftup = selfwall_coords[0][0]
        leftupx = selfwall_coords[0][0][0]
        leftupy = selfwall_coords[0][0][1]
        rightdown = selfwall_coords[0][1]
        rightdownx = selfwall_coords[0][1][0]
        rightdowny = selfwall_coords[0][1][1]
        old_state=s
        #print('old_state',old_state)
        if (old_state <= leftup).all():  # left upper#old_state#check it!
            #reldistold = old_state - leftup  # relative distance old#np.linalg.norm()
            #print('enter region 1! time: ',t)
            AXE=np.array([[1,1]])
            BXE=np.array([leftupx+leftupy-15-goal[0]-goal[1]])
            u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin, AXE, BXE)  # use P_lyap
        elif leftupx <= old_state[0] <= rightdownx and old_state[1] <= leftupy:
            #reldistold = np.array([0, old_state[1] - leftupy])  # middle up
            #print('enter region 2! time: ', t)
            xmax[1]=40
            u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin)  # use P_lyap
        elif old_state[0] >= rightdownx and old_state[1] <= leftupy:
            #reldistold = old_state - (rightdownx, leftupy)  # upper right
            #print('enter region 3! time: ', t)
            AXE = np.array([[-1, 1]])
            BXE = np.array([-(rightdownx-leftupy+15-goal[0]+goal[1])])
            u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin, AXE, BXE)  # use P_lyap
        elif old_state[0] >= rightdownx and leftupy <= old_state[1] <= rightdowny:
            #reldistold = np.array([old_state[0] - rightdownx, 0])  # right middle
            #print('enter region 4! time: ', t)
            xmin[0] = 115
            u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin)  # use P_lyap
        elif (old_state >= rightdown).all():  # old_state#lower right
            #reldistold = old_state - rightdown
            #print('enter region 5! time: ', t)
            AXE = np.array([[-1, -1]])
            BXE = np.array([-(rightdownx+leftupy+15-goal[0]-goal[1])])
            u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin, AXE, BXE)  # use P_lyap
        elif leftupx <= old_state[0] <= rightdownx and old_state[1] >= rightdowny:
            #reldistold = np.array([0, old_state[1] - rightdowny])  # middle down/lower middle
            #print('enter region 6! time: ', t)
            xmin[1] = 110
            u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin)  # use P_lyap
        elif old_state[0] <= leftupx and old_state[1] >= rightdowny:
            #reldistold = (old_state - (leftupx, rightdowny))  # lower left
            #print('enter region 7! time: ', t)
            AXE = np.array([[1, -1]])
            BXE = np.array([-(rightdowny-leftupx+15+goal[0]-goal[1])])
            u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin, AXE, BXE)  # use P_lyap
        elif old_state[0] <= leftupx and leftupy <= old_state[1] <= rightdowny:
            #reldistold = np.array([old_state[0] - leftupx, 0])  # middle left
            #print('enter region 8! time: ', t)
            xmax[0] = 60
            #print('xmax',xmax)
            u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin)  # use P_lyap
        #else:
            #print('enter region obstacle! time: ', t)
            # print(old_state)#it can be [98.01472841 92.11425524]
            #reldistold = np.array([0, 0])  # 9.9#
        #u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin,AXE,BXE)  # use P_lyap
        UK = u0[0]  # FKI*XK%pick the first one from u0 to implement
        act=UK
        act = np.clip(act, -um, um)
        return act#pay attention to its dimension

class ConstraintTeacher(AbstractTeacher):
    def __init__(self, env, noisy=True):
        super().__init__(env, noisy, on_policy=False)
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE
        self.goal = (88, 75)#within the obstacle! lead to collision!
        self.random_start = True

    def _expert_control(self, state, i):
        if i < 15:#as said in the paper, random action
            return self.d
        else:
            to_obstactle = np.subtract(self.goal, state)
            to_obstacle_normalized = to_obstactle / np.linalg.norm(to_obstactle)#direction
            to_obstactle_scaled = to_obstacle_normalized * spb.MAX_FORCE / 2
            return to_obstactle_scaled

    def reset(self):
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE

class ConstraintTeacherMPC(AbstractTeacher):
    def __init__(self, env, noisy=True):
        super().__init__(env, noisy, on_policy=False)
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE
        self.goal = (88, 75)#within the obstacle! lead to collision!
        self.random_start = True

    def _expert_control(self, state, i):
        if i < 15:#as said in the paper, random action
            return self.d
        else:
            #to_obstactle = np.subtract(self.goal, state)
            A = np.array([[1, 0], [0, 1]])  # [EV1,EV2]=linalg.eigvals(A)#ACTUALLY NOT USED
            B = np.array([[1, 0], [0, 1]])  # ([[0.2173],[0.0573]])#what should the dim of B be?
            QV1 = 1  #100  #  ;#500;#change this in part 8#QV for Q value
            QV2 = 1  #100  #  500;#;#change this in part 8#
            Q = np.diag([QV1, QV2])  # QV.*eye(2);%100*eye(2);#
            RV1 = 1;RV2 = 1
            R = np.diag([RV1, RV2])  # R
            N = 5  # 1;#2;#25#10#;#prediction horizon, change this in part 6,7,8#
            PRIC = la.solve_discrete_are(A, B, Q, R)  # P_ric
            width = 180  # bignumber = 100000000  # as if it is infinity
            height = 150
            xmin = [-1, -1]  # [-bignumber,-1]#0]#0#-100000#-inf
            xmax = [+width, +height]  # 10000000000#inf
            um = spb.MAX_FORCE
            umin = [-um, -um]  # -5#-50#-100#
            umax = [um, um]  # 5#50#100#
            x0 = state.reshape((state.shape[0],1))#np.array([[6], [60]])  # [0;10]#x0 is the initial condition, will be changed later#
            goal=np.asarray(self.goal)
            xdestin = goal.reshape((goal.shape[0], 1))#array or not?  # np.array([[2], [2]])  #
            # for i in range(int(N2 * (1 / T))):  # u0=mympc(A,B,Q,R,PRIC,N,umin,umax,xmin,xmax,XK)%use P_ric
            u0 = mympc(A, B, Q, R, PRIC, N, umin, umax, xmin, xmax, x0, xdestin)  # use P_lyap
            UK = u0[0]  # FKI*XK%pick the first one from u0 to implement
            to_obstactle = UK
            to_obstacle_normalized = to_obstactle / np.linalg.norm(to_obstactle)#direction
            to_obstactle_scaled = to_obstacle_normalized * spb.MAX_FORCE / 2
            return to_obstactle_scaled

    def reset(self):
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE

class ConstraintTeachersymmetric(AbstractTeacher):
    def __init__(self, env, noisy=True):
        super().__init__(env, noisy, on_policy=False)
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE
        self.goal = (88, 75)#within the obstacle! lead to collision!
        self.random_start = True
        self.deven=self.d

    def _expert_control(self, state, i):
        if i < 15:#as said in the paper, random action
            return self.d
        else:
            to_obstactle = np.subtract(self.goal, state)
            to_obstacle_normalized = to_obstactle / np.linalg.norm(to_obstactle)#direction
            to_obstactle_scaled = to_obstacle_normalized * spb.MAX_FORCE / 2
            return to_obstactle_scaled

    def reset(self):
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE

class ReacherTeacher(AbstractTeacher):
    def __init__(self, env, noisy=True):
        super().__init__(env, noisy=noisy, horizon=100)

    def _expert_control(self, state, i):
        if i < 40:
            goal = np.array((np.pi, 0))
        else:
            goal = np.array((np.pi * .75, 0))

        angle = state[:2]
        act = goal - angle
        act = np.clip(act, -1, 1)
        return act


class ReacherConstraintTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super(ReacherConstraintTeacher, self).__init__(env, noisy, on_policy=False)
        self.direction = 1
        self.random_start = True

    def _expert_control(self, state, i):
        angle = state[:2]
        goal1 = np.array((np.pi * .53, 0.7 * np.pi))
        goal2 = np.array((np.pi, -0.7 * np.pi))
        goal = min(goal1, goal2, key=lambda x: np.linalg.norm(angle - x))
        act = goal - angle
        # act = np.random.normal((self.direction, 0), 1)
        act = np.clip(act, -1, 1)
        return act

    def reset(self):
        self.direction = self.direction * -1


class PushTeacher(AbstractTeacher):

    def __init__(self, env, noisy):
        super(PushTeacher, self).__init__(env, False)
        self.demonstrations = []
        self.default_noise = 0.2
        self.block_id = 0
        self.horizon = 150

    def _expert_control(self, state, i):
        action, block_done = self.env.expert_action(block=self.block_id, noise_std=0.004)
        if block_done:
            self.block_id += 1
            self.block_id = min(self.block_id, 2)

        return action

    def reset(self):
        self.block_id = 0


class StrangeTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super(StrangeTeacher, self).__init__(env, noisy, on_policy=False)
        self.d_act = env.action_space.shape
        self.high = env.action_space.high
        self.low = env.action_space.low
        self.std = (self.high - self.low) / 10
        self.last_action = env.action_space.sample()
        self.random_start = True
        self.horizon = 20

    def _expert_control(self, state, i):
        action = np.random.normal(self.last_action, self.std)
        action = np.clip(action, self.low, self.high)
        self.last_action = action
        return action

    def reset(self):
        self.last_action = self.env.action_space.sample()


class OutburstPushTeacher(AbstractTeacher):
    def __init__(self, env, noisy):
        super(OutburstPushTeacher, self).__init__(env, False, False)
        # self.block_id = 0
        self.horizon = 150
        self.outburst = False

    def _expert_control(self, state, i):
        if np.random.random() > .8:
            self.outburst = True

        if np.random.random() > .9:
            self.outburst = False

        if self.outburst:
            return self.env.action_space.sample().astype(np.float64)

        return np.array((0, -0.02))

    def reset(self):
        self.block_id = 0
        self.outburst = False



