from abc import ABC

from latentsafesets.envs import simple_point_bot as spb

import numpy as np


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
        self.reset()#for pushing, it is just set the block_id=0
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
                action_input = np.clip(action_input, self.ac_low + 1e-6, self.ac_high - 1e-6)
            else:
                action_input = action

            if store_noisy:
                action = action_input#if it not noisy, then it is just the same
            #import ipdb; ipdb.set_trace()
            #print('action',action)
            #print('action_input',action_input)
            #print('actiond',action.dtype)#float64, not np.float64!
            #print('action_inputs',action_input.shape)
            action_input=np.float32(action_input)#has to be like this?
            #action=np.float32(action)#has to be like this? maybe it takes float 64, but not takes float 32 as the float
            #print('actiond2',action.dtype)#float64, not np.float64!
            #print('action_inputs2',action_input.shape)
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
        obs = self.env.reset(random_start=self.random_start)    #obs is a 3 channel image!
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
                #action_input = np.clip(action_input, self.ac_low + 1e-6, self.ac_high - 1e-6)
            else:
                action_input = action

            if store_noisy:
                action = action_input#if it not noisy, then it is just the same
            #import ipdb; ipdb.set_trace()
            action_input=np.float32(action_input)#has to be like this?#this is important!
            #next_obs, reward, done, info = self.env.step(action_input)#for reacher#63 in simple_point_bot.py
            next_obs, reward, done, info = self.env.stepsafety(action_input)#for pushing and for spb#63 in simple_point_bot.py
            transition = {'obs': obs, 'action': tuple(action), 'reward': float(reward),
                          'next_obs': next_obs, 'done': int(done),#this is a dictionary
                          'constraint': int(info['constraint']), 'safe_set': 0,
                          'on_policy': int(self.on_policy),##
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
        #print('transitions[obs]',transitions[0]['obs'])#it looks normal
        # assert done, "Did not reach the goal set on task completion."
        # V = self.env.values()
        # for i, t in enumerate(transitions):
        #     t['values'] = V[i]
        return transitions

    def generate_trajectorysafety_relative(self, noise_param=None, store_noisy=True):
        """
        The teacher initially tries to go northeast before going to the origin
        """
        self.reset()
        transitions = []#AN EMPTY LIST
        obs,obs_relative = self.env.reset(random_start=self.random_start)#still in global state#obs is a 3 channel image!#around line 85 in simple_point_bot.py#random_start is false by default
        #print('obs_relative',obs_relative)
        state = None# state = np.zeros((0, 0))
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
            #next_obs, reward, done, info,next_obs_relaitve = self.env.stepsafety(action_input)#63 in simple_point_bot.py
            next_obs, reward, done, info, next_obs_relative = self.env.stepsafety_relative(
                action_input)  # 63 in simple_point_bot.py
            transition = {'obs': obs, 'action': tuple(action), 'reward': float(reward),#that obs should correspond to hvo
                          'next_obs': next_obs, 'done': int(done),#this is a dictionary
                          'constraint': int(info['constraint']), 'safe_set': 0,
                          'on_policy': int(self.on_policy),
                          'rdo': info['rdo'].tolist(),
                          'rdn': info['rdn'].tolist(),
                          'hvo': info['hvo'],#obs here
                          'hvn': info['hvn'],#next_obs here
                          'hvd': info['hvd'],
                          'state':info['state'].tolist(),
                          'next_state':info['next_state'].tolist(),
                          'state_relative': info['state_relative'].tolist(),
                          'next_state_relative': info['next_state_relative'].tolist(),
                          'obs_relative':obs_relative,
                          'next_obs_relative':next_obs_relative
                          }#add key and value into it!
            # print({k: v.dtype for k, v in transition.items() if 'obs' in k})
            transitions.append(transition)#a list of dictionaries!
            #print("transition['obs'].shape",transition['obs'].shape)
            #print("transition['obs_relative'].shape", transition['obs_relative'].shape)
            #print("transition['next_obs'].shape", transition['next_obs'].shape)
            #print("transition['next_state_relative'].shape", transition['next_obs_relative'].shape)
            state = info['next_state']
            obs = next_obs#this step is really important!
            obs_relative = next_obs_relative

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
        act = np.clip(act, -1+1e-6, 1-1e-6)#np.clip(act, -1, 1)# + 1e-6
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
        action, block_done = self.env.expert_action(block=self.block_id, noise_std=0.004)#the action we get is with noise!
        if block_done:
            self.block_id += 1
            self.block_id = min(self.block_id, 2)#3 blocks: 0, 1, 2. everytime focus on one block!

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
        if np.random.random() > .8:#only 10% random actions?
            self.outburst = True

        if np.random.random() > .9:
            self.outburst = False

        if self.outburst:
            return self.env.action_space.sample().astype(np.float64)

        return np.array((0, -0.02))#-0.02 means it is pushing downwards

    def reset(self):
        self.block_id = 0
        self.outburst = False



