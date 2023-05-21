'''
Built on cartrgipper implementation in
https://github.com/SudeepDasari/visual_foresight
'''

import numpy as np
import moviepy.editor as mpy
import copy
from .base_mujoco_env import BaseMujocoEnv
from gym.spaces import Box
import os

FIXED_ENV = True
GT_STATE = False
EARLY_TERMINATION = False


def no_rot_dynamics(prev_target_qpos, action):#? what's this for?
    target_qpos = np.zeros_like(prev_target_qpos)
    target_qpos[:3] = action[:3] + prev_target_qpos[:3]
    target_qpos[4] = action[3]#should this be 3 or 4?
    return target_qpos


def clip_target_qpos(target, lb, ub):
    target[:len(lb)] = np.clip(target[:len(lb)], lb, ub)
    return target


class PushEnv(BaseMujocoEnv):
    def __init__(self):
        parent_params = super()._default_hparams()#It's calling the _default_hparams() from its parent, BaseMujocoEnv
        envs_folder = os.path.dirname(os.path.abspath(__file__))#this is only for the next line!
        self.reset_xml = os.path.join(envs_folder, 'push_env.xml')#I know this!
        super().__init__(self.reset_xml, parent_params)#This's calling the __init__() from its parent, BaseMujocoEnv
        self._adim = 2
        self.substeps = 500
        self.low_bound = np.array([-0.4, -0.4, -0.05])#what's this bound for?
        self.high_bound = np.array([0.4, 0.4, 0.15])
        self.ac_high = 0.05*np.ones(self._adim)#this is the action bound, right?
        self.ac_low = -self.ac_high
        self.action_space = Box(self.ac_low, self.ac_high)
        self._previous_target_qpos = None
        self.target_height_thresh = 0.03
        self.object_fall_thresh = -0.03
        self.obj_y_dist_range = np.array([0.05, 0.05])
        self.obj_x_range = np.array([-0.03, -0.03])
        self.randomize_objects = not FIXED_ENV
        self.gt_state = GT_STATE
        self._max_episode_steps = 150
        self.horizon = 150
        self._num_steps = 0
        # self._viol = False

        if self.gt_state:#if states can be acquired!
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(27, ))#all those states
        else:#it is the image inputs, all those states combined in the image
            self.observation_space = self.observation_space = Box(0, 1, shape=(3, 64, 64), dtype='float32')
        self.reset()
        self.num_objs = (self.position.shape[0] - 6) // 7#???

    def render(self):#render means generating images
        x = super().render()[:, ::-1].copy().squeeze()#this is using the render method from its parents!
        return x.transpose((2, 0, 1)) # cartgripper cameras are flipped in height dimension#some data processing?

    def reset(self, **kwargs):#that is only the reset!
        self._reset_sim(self.reset_xml)#see the parent! It is directly using the method from the parent, BaseMujocoEnv
        #clear our observations from last rollout
        self._last_obs = None#that's why we start from everything being put together, still the method from parent

        state = self.sim.get_state()#where is this? See its parents!
        pos = np.copy(state.qpos[:])#what is this qpos?
        pos[6:] = self.object_reset_poses().ravel()#around 257#ravel means changing into 1 dimension so it goes from (3,7) to 21
        state.qpos[:] = pos#also, the above object_reset_pose means that, at the first/0th frame, it is everything hold together
        self.sim.set_state(state)
        self._num_steps = 0
        # self._viol = False

        self.sim.forward()#self.sim is the MjSim shown in the parent

        self._previous_target_qpos = copy.deepcopy(
            self.sim.data.qpos[:5].squeeze())
        self._previous_target_qpos[-1] = self.low_bound[-1]

        if self.gt_state:
            return pos
        else:
            return self.render()#around 63

    def step(self, action):
        position = self.position
        # print("ACTION: ", action)
        action = np.clip(action, self.ac_low, self.ac_high)
        # Add extra action dimensions that we have artificially removed
        action = np.array(list(action) + [0, 0])
        target_qpos = self._next_qpos(action)#297
        if self._previous_target_qpos is None:#with reset, it should not be none?
            self._previous_target_qpos = target_qpos
        finger_force = np.zeros(2)

        for st in range(self.substeps):#500
            alpha = st / (float(self.substeps) - 1)
            self.sim.data.ctrl[:] = alpha * target_qpos + (
                1. - alpha) * self._previous_target_qpos
            self.sim.step()#self.sim is the MjSim thing!

        self._previous_target_qpos = target_qpos
        # constraint = self._viol or self.topple_check()
        # self._viol = constraint
        constraint = self.constraint_fn()#246
        reward = self.reward_fn()#253
        if constraint:
            reward = -1

        self._num_steps += 1
        if EARLY_TERMINATION:
            done = (constraint > 0) or (reward > -0.5)
        else:
            done = self._num_steps >= self._max_episode_steps

        info = {
            "constraint": constraint,
            "reward": reward,
            "state": position,#the position of the end effector?
            "next_state": self.position,#it changes along the way?
            "action": action
        }
        #print('state',position,'next_state',self.position)
        if self.gt_state:
            return self.position, reward, done, info
        else:
            return self.render(), reward, done, info#self.render() seems to generate the image?

    def topple_check(self, debug=False):#seems not being used?
        quat = self.object_poses[:, 3:]
        phi = np.arctan2(
            2 *
            (np.multiply(quat[:, 0], quat[:, 1]) + quat[:, 2] * quat[:, 3]),
            1 - 2 * (np.power(quat[:, 1], 2) + np.power(quat[:, 2], 2)))
        theta = np.arcsin(2 * (np.multiply(quat[:, 0], quat[:, 2]) -
                               np.multiply(quat[:, 3], quat[:, 1])))
        psi = np.arctan2(
            2 * (np.multiply(quat[:, 0], quat[:, 3]) + np.multiply(
                quat[:, 1], quat[:, 2])),
            1 - 2 * (np.power(quat[:, 2], 2) + np.power(quat[:, 3], 2)))
        euler = np.stack([phi, theta, psi]).T[:, :2] * 180. / np.pi
        if debug:
            return np.abs(euler).max() > 15 or np.isnan(euler).sum() > 0, euler
        return np.abs(euler).max() > 15 or np.isnan(euler).sum() > 0

    @property
    def jaw_width(self):
        pos = self.position
        return 0.08 - (pos[4] - pos[5])  # 0.11 might be slightly off

    def set_y_range(self, bounds):
        self.obj_y_dist_range[0] = bounds[0]
        self.obj_y_dist_range[1] = bounds[1]

    def expert_action(self, block=0, noise_std=0.001, step_size=0.05):
        # Can make step_size smaller to make demos more slow
        # Can also increase noise_std but this may make demos less reliable
        # print("BLOCK ID: ", block)
        cur_pos = self.position[:3]#this is the position of the end effector
        cur_pos[1] += 0.05  # compensate for length of jaws#a fixed bias?

        block_done = False
        block_reset_done = False
        block_pos = self.object_poses[block][:3]#0,1,2#those are object poses, not pose of the end effector
        action = np.zeros(self._adim)
        delta = block_pos - cur_pos#3 dimensional vector
        if not block_done:
            if abs(delta[0]) > 1e-3:#that is the y, the movement in the horizontal direction!
                action[0] = delta[0]#which will be confined by the constraint of the action limit
            else:
                if abs(block_pos[1]) < 0.1:#it should be the x position, or the row/vertical position
                    action[1] = -step_size#abs(block_pos)>0.075 will get you done for that block!
                else:
                    block_done = True
        if block_done:
            if cur_pos[1] < 0.04:#if the current pose for the jaw is not back enough
                action[1] = step_size#?what's this for?#retreat back!
            else:
                block_reset_done = True
  
        action = action + np.random.randn(self._adim) * noise_std
        action = np.clip(action, self.ac_low, self.ac_high)#the action really implemented is with noise!!!
        return action, block_reset_done

    def get_demo(self, noise_std=0.001):
        im_list = []
        obs_list = [self.reset()]
        ac_list = []
        block_id = 0
        num_steps = 0
        while block_id < self.num_objs and num_steps < self._max_episode_steps:#150
            ac, reset_done = self.expert_action(block=block_id, noise_std=noise_std)
            if reset_done:
                block_id += 1
            ns, r, done, info = self.step(ac)
            obs_list.append(ns)
            ac_list.append(ac)
            im_list.append(self.render().squeeze())
            num_steps += 1

        while num_steps < self._max_episode_steps:
            ac = 5 * np.random.randn(self._adim) * noise_std
            ns, r, done, info = self.step(ac)
            obs_list.append(ns)
            ac_list.append(ac)
            im_list.append(self.render().squeeze())
            num_steps += 1

        # npy_to_gif(im_list, "out") # vis stuff for debugging
        return obs_list, ac_list, im_list

    def get_rand_rollout(self):
        if np.random.random() < 0.6: # TODO: may need to tune this
            obs_list, ac_list, im_list = self.get_demo(noise_std=0.01)
        else:
            im_list = []
            obs_list = [self.reset()]
            ac_list = []
            num_steps = 0
            while num_steps < self._max_episode_steps:
                ac = self.action_space.sample()#random action
                ns, r, done, info = self.step(ac)
                obs_list.append(ns)
                ac_list.append(ac)
                im_list.append(self.render().squeeze())
                num_steps += 1

        npy_to_gif(im_list, "out") # vis stuff for debugging
        return obs_list, ac_list, im_list

    def get_block_dones(self):
        block_dones = np.zeros(self.num_objs)
        for block in range(self.num_objs):
            block_pos = self.object_poses[block][:3]
             # TODO: maybe make this an interval rather than a threshold later
            if 0.075 < abs(block_pos[1]) and abs(block_pos[0]) < .2:#x>0.075, y<0.2? what is this?
                block_dones[block] = 1
        return block_dones

    def constraint_fn(self):
        block_constr = []
        for block in range(self.num_objs):
            block_pos = self.object_poses[block][:3]
            block_constr.append(abs(block_pos[2]) > .2)#high or low? Falling means low!
        return any(block_constr)

    def reward_fn(self):
        block_dones = self.get_block_dones()
        return int(np.sum(block_dones) == len(block_dones)) - 1#this means all 3 blocks must be pushed enough to get reward

    def object_reset_poses(self):
        new_poses = np.zeros((3, 7))
        new_poses[:, 3] = 1#what is this?
        if self.randomize_objects == True:
            x = np.random.uniform(self.obj_x_range[0], self.obj_x_range[1])
            y1 = np.random.randn() * 0.05
            y0 = y1 - np.random.uniform(self.obj_y_dist_range[0],#0.05
                                        self.obj_y_dist_range[1])
            y2 = y1 + np.random.uniform(self.obj_y_dist_range[0],
                                        self.obj_y_dist_range[1])
            new_poses[0, 0:2] = np.array([y0, x])#new pose 0th column is y, 1th column is x!
            new_poses[1, 0:2] = np.array([y1, x])
            new_poses[2, 0:2] = np.array([y2, x])
        else:
            x = np.mean(self.obj_x_range)#which is -0.03
            y1 = 0.
            y0 = y1 - np.mean(self.obj_y_dist_range)
            y2 = y1 + np.mean(self.obj_y_dist_range)
            new_poses[0, 0:2] = np.array([y0, x])#what is x or y? row or column?
            new_poses[1, 0:2] = np.array([y1, x])
            new_poses[2, 0:2] = np.array([y2, x])
        return new_poses

    @property
    def position(self):#here it is 27 dimensional!
        return np.copy(self.sim.get_state().qpos[:])

    @property
    def object_poses(self):
        pos = self.position
        num_objs = (self.position.shape[0] - 6) // 7
        poses = []
        for i in range(num_objs):
            poses.append(np.copy(pos[i * 7 + 6:(i + 1) * 7 + 6]))
        return np.array(poses)#object poses, not end effector poses

    @property
    def target_object_height(self):
        return self.object_poses[1, 2] - 0.072

    def _next_qpos(self, action):
        target = no_rot_dynamics(self._previous_target_qpos, action)
        target = clip_target_qpos(target, self.low_bound, self.high_bound)
        return target

    def stepsafety(self, action):
        position = self.position#this is the 27 dimensional thing!
        # print("ACTION: ", action)
        oldefy=position[0]#oldefy means old end effector y coordinate
        oldblock1y=position[6]#
        oldblock2y=position[13]
        oldblock3y=position[20]
        #some min max things
        oefyabs=np.abs(oldefy)
        ob1yabs=np.abs(oldblock1y)#old block 1 y abs 
        ob2yabs=np.abs(oldblock2y)
        ob3yabs=np.abs(oldblock3y)
        #ob1yasat=np.where(ob1yabs<=0.4,ob1yabs,0)#old block 1 y abs saturate
        #ob2yasat=np.where(ob2yabs<=0.4,ob2yabs,0)
        #ob3yasat=np.where(ob3yabs<=0.4,ob3yabs,0)
        thres=0.3
        obf1=thres**2-ob1yabs**2#-ob1yasat**2#
        obf2=thres**2-ob2yabs**2#-ob2yasat**2#
        obf3=thres**2-ob3yabs**2#-ob3yasat**2#
        rdo=max(ob1yabs,ob2yabs,ob3yabs)#max(ob1yasat,ob2yasat,ob3yasat)#
        hvo=min(obf1,obf2,obf3)#the more negative, the more unsafe
        action = np.clip(action, self.ac_low, self.ac_high)
        # Add extra action dimensions that we have artificially removed
        action = np.array(list(action) + [0, 0])
        target_qpos = self._next_qpos(action)#297
        if self._previous_target_qpos is None:#with reset, it should not be none?
            self._previous_target_qpos = target_qpos
        finger_force = np.zeros(2)

        for st in range(self.substeps):#500
            alpha = st / (float(self.substeps) - 1)
            self.sim.data.ctrl[:] = alpha * target_qpos + (
                1. - alpha) * self._previous_target_qpos
            self.sim.step()#self.sim is the MjSim thing!

        self._previous_target_qpos = target_qpos
        # constraint = self._viol or self.topple_check()
        # self._viol = constraint
        constraint = self.constraint_fn()#246
        reward = self.reward_fn()#253
        if constraint:
            reward = -1

        self._num_steps += 1
        if EARLY_TERMINATION:
            done = (constraint > 0) or (reward > -0.5)
        else:
            done = self._num_steps >= self._max_episode_steps

        newefy=self.position[0]#oldefy means old end effector y coordinate
        newblock1y=self.position[6]#
        newblock2y=self.position[13]
        newblock3y=self.position[20]
        #some min max things
        nefyabs=np.abs(newefy)
        nb1yabs=np.abs(newblock1y)#old block 1 y abs 
        nb2yabs=np.abs(newblock2y)
        nb3yabs=np.abs(newblock3y)
        #nb1yasat=np.where((nb1yabs<=0.4)|((nb1yabs>0.4)&(ob1yabs<=0.4)),nb1yabs,0)#old block 1 y abs saturate
        #nb2yasat=np.where((nb2yabs<=0.4)|((nb2yabs>0.4)&(ob2yabs<=0.4)),nb2yabs,0)#try to capture that moment!
        #nb3yasat=np.where((nb3yabs<=0.4)|((nb3yabs>0.4)&(ob3yabs<=0.4)),nb3yabs,0)#the most ideal way!
        thres=0.3
        nbf1=thres**2-nb1yabs**2#-nb1yasat**2#
        nbf2=thres**2-nb2yabs**2#-nb2yasat**2#
        nbf3=thres**2-nb3yabs**2#-nb3yasat**2#
        rdn=max(nb1yabs,nb2yabs,nb3yabs)#max(nb1yasat,nb2yasat,nb3yasat)#
        hvn=min(nbf1,nbf2,nbf3)#the more negative, the more unsafe
        hvd=hvn-hvo

        info = {
            "constraint": constraint,
            "reward": reward,
            "state": position,#the position of the end effector? the 27 dimensional thing!
            "next_state": self.position,#it changes along the way?
            "action": action,
            "rdo":rdo,#rdo for relative distance old#array now!
            "rdn": rdn,#rdn for relative distance new#array now!
            "hvo": hvo,#hvo for h value old#corresponding to the old state
            "hvn":hvn,#hvn for h value new#corresponding to the new state
            "hvd":hvd #hvd for h value difference
        }
        #print('state',position,'next_state',self.position)
        if self.gt_state:
            return self.position, reward, done, info
        else:
            return self.render(), reward, done, info#self.render() seems to generate the image?
    def stepsafety2(self, action):#push strategy 2!
        position = self.position#this is the 27 dimensional thing!
        # print("ACTION: ", action)
        oldefy=position[0]#oldefy means old end effector y coordinate#confirmed it is the y position!!!
        oldblock1y=position[6]#
        oldblock2y=position[13]
        oldblock3y=position[20]
        #some min max things
        oefyabs=np.abs(oldefy)
        ob1yabs=np.abs(oldblock1y)#old block 1 y abs 
        ob2yabs=np.abs(oldblock2y)
        ob3yabs=np.abs(oldblock3y)
        #ob1yasat=np.where(ob1yabs<=0.4,ob1yabs,0)#old block 1 y abs saturate
        #ob2yasat=np.where(ob2yabs<=0.4,ob2yabs,0)
        #ob3yasat=np.where(ob3yabs<=0.4,ob3yabs,0)
        thres=0.3
        thresef=0.2#3
        oeff=thresef**2-oefyabs**2#old end effector function value
        obf1=thres**2-ob1yabs**2#-ob1yasat**2#
        obf2=thres**2-ob2yabs**2#-ob2yasat**2#
        obf3=thres**2-ob3yabs**2#-ob3yasat**2#
        rdo=max(ob1yabs,ob2yabs,ob3yabs)#max(ob1yasat,ob2yasat,ob3yasat)#
        hvo=min(obf1,obf2,obf3)#the more negative, the more unsafe
        rdoef=oefyabs#max(ob1yabs,ob2yabs,ob3yabs)#max(ob1yasat,ob2yasat,ob3yasat)#
        hvoef=oeff#min(obf1,obf2,obf3)#the more negative, the more unsafe
        action = np.clip(action, self.ac_low, self.ac_high)
        # Add extra action dimensions that we have artificially removed
        action = np.array(list(action) + [0, 0])
        target_qpos = self._next_qpos(action)#297
        if self._previous_target_qpos is None:#with reset, it should not be none?
            self._previous_target_qpos = target_qpos
        finger_force = np.zeros(2)

        for st in range(self.substeps):#500
            alpha = st / (float(self.substeps) - 1)
            self.sim.data.ctrl[:] = alpha * target_qpos + (
                1. - alpha) * self._previous_target_qpos
            self.sim.step()#self.sim is the MjSim thing!

        self._previous_target_qpos = target_qpos
        # constraint = self._viol or self.topple_check()
        # self._viol = constraint
        constraint = self.constraint_fn()#246
        reward = self.reward_fn()#253
        if constraint:
            reward = -1

        self._num_steps += 1
        if EARLY_TERMINATION:
            done = (constraint > 0) or (reward > -0.5)
        else:
            done = self._num_steps >= self._max_episode_steps

        newefy=self.position[0]#oldefy means old end effector y coordinate
        newblock1y=self.position[6]#
        newblock2y=self.position[13]
        newblock3y=self.position[20]
        #some min max things
        nefyabs=np.abs(newefy)
        nb1yabs=np.abs(newblock1y)#old block 1 y abs 
        nb2yabs=np.abs(newblock2y)
        nb3yabs=np.abs(newblock3y)
        #nb1yasat=np.where((nb1yabs<=0.4)|((nb1yabs>0.4)&(ob1yabs<=0.4)),nb1yabs,0)#old block 1 y abs saturate
        #nb2yasat=np.where((nb2yabs<=0.4)|((nb2yabs>0.4)&(ob2yabs<=0.4)),nb2yabs,0)#try to capture that moment!
        #nb3yasat=np.where((nb3yabs<=0.4)|((nb3yabs>0.4)&(ob3yabs<=0.4)),nb3yabs,0)#the most ideal way!
        neff=thresef**2-nefyabs**2#new end effector function value
        nbf1=thres**2-nb1yabs**2#-nb1yasat**2#
        nbf2=thres**2-nb2yabs**2#-nb2yasat**2#
        nbf3=thres**2-nb3yabs**2#-nb3yasat**2#
        rdn=max(nb1yabs,nb2yabs,nb3yabs)#max(nb1yasat,nb2yasat,nb3yasat)#
        hvn=min(nbf1,nbf2,nbf3)#the more negative, the more unsafe
        rdnef=nefyabs#max(ob1yabs,ob2yabs,ob3yabs)#max(ob1yasat,ob2yasat,ob3yasat)#
        hvnef=neff#min(obf1,obf2,obf3)#the more negative, the more unsafe
        hvd=hvn-hvo
        hvdef=hvnef-hvoef

        info = {
            "constraint": constraint,
            "reward": reward,
            "state": position,#the position of the end effector?
            "next_state": self.position,#it changes along the way?
            "action": action,
            "rdo":rdo,#rdo for relative distance old#array now!
            "rdn": rdn,#rdn for relative distance new#array now!
            "hvo": hvo,#hvo for h value old#corresponding to the old state
            "hvn":hvn,#hvn for h value new#corresponding to the new state
            "hvd":hvd, #hvd for h value difference
            "rdoef":rdoef,#rdo for relative distance old#array now!
            "rdnef": rdnef,#rdn for relative distance new#array now!
            "hvoef": hvoef,#hvo for h value old#corresponding to the old state
            "hvnef":hvnef,#hvn for h value new#corresponding to the new state
            "hvdef":hvdef #hvd for h value difference
        }
        #print('state',position,'next_state',self.position)
        if self.gt_state:
            return self.position, reward, done, info
        else:
            return self.render(), reward, done, info#self.render() seems to generate the image?


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


if __name__ == '__main__':
    env = PushEnv()
    # env.get_demo()
    env.get_rand_rollout()
