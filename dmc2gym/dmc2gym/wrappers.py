from gym import core, spaces
from dm_control import suite
#from ../../dm_control.dm_control import suite
#from dm_control.dm_control import suite
from dm_env import specs
import numpy as np


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:#it seems that the reacher environment doesn't have action constraint
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)#those values from those keys


class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first
        self._frozen = False
        self._n_steps = 0
        self.horizon = 100

        # create task
        self._env = suite.load(
            domain_name=domain_name,#for example, if I want to load reach, then this domain_name will be reacher
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])#see control.Environment
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )
            
        self._state_space = _spec_to_box(
                self._env.observation_spec().values()
        )
        
        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(#render an image
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )#get the pictures!
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def get_image_obs(self):
        return self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            ).copy()

    def _convert_action(self, action):
        action = action.astype(np.float64)
        action = action / 2
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)
    
    def step(self, action):#the old original step
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = -1 * self._frame_skip#print('running this customized version!')
        #print('enter this step in trajectory generation!')#yes, it enters this step as desired!
        old_observation = _flatten_obs(self._env._task.get_observation(self._env.physics))#what does it contain?#4 rows, 2 columns
        old_constraint = self._env._task.get_constraint(self._env.physics)#I assume the constraint is got from the customized reacher.py
        if self._frozen:
            action = np.zeros_like(action)

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)#propogate
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        observation = _flatten_obs(self._env._task.get_observation(self._env.physics))
        constraint = self._env._task.get_constraint(self._env.physics)
        if constraint:
            self._frozen = True
        # reward -= constraint * 100
        self._n_steps += 1
        # print(self._n_steps)
        done = self._n_steps >= 100

        # TODO: [redacted name :)] update this because these envs are force controlled
        if constraint and not old_constraint:
            constraint_cost = np.linalg.norm(action)
        else:
            constraint_cost = 0

        extra = {'internal_state': self._env.physics.get_state().copy(),#it seems that you can freely add key-value pairs in extra!
                 'discount': time_step.discount,#not show up in spb!
                 'constraint': constraint,
                 'constraint_cost': constraint_cost,
                 'reward': reward,
                 'state': old_observation,#it containts the 2 dimensional vector to the center of the obstacle as its 3rd/2nd row!
                 'next_state': observation,
                 'action': action}

        return obs, reward, done, extra#that's how it originally works!
    '''
    def step(self, action):#originally it was called step safety
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = -1 * self._frame_skip#print('running this customized version!')
        #print('enter this stepsafety in trajectory generation!')#I know it is entering it as expected!
        old_observation = _flatten_obs(self._env._task.get_observation(self._env.physics))#what does it contain?
        old_constraint = self._env._task.get_constraint(self._env.physics)#I assume the constraint is got from the customized reacher.py
        if self._frozen:
            action = np.zeros_like(action)

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)#propogate
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        observation = _flatten_obs(self._env._task.get_observation(self._env.physics))
        #print('old_observation',old_observation,'observation',observation)
        constraint = self._env._task.get_constraint(self._env.physics)
        if constraint:
            self._frozen = True
        # reward -= constraint * 100
        self._n_steps += 1
        # print(self._n_steps)
        done = self._n_steps >= 100

        # TODO: [redacted name :)] update this because these envs are force controlled
        if constraint and not old_constraint:
            constraint_cost = np.linalg.norm(action)
        else:
            constraint_cost = 0

        oldposition=old_observation[0:2]
        oldtotarget=old_observation[2:4]    
        oldtoobstacle=old_observation[4:6]
        oldvelocity=old_observation[6:8]
        #print('oldposition',oldposition,'oldtotarget',oldtotarget,'oldtoobstacle',oldtoobstacle,'oldvelocity',oldvelocity)
        oldtoonorm=oldtoobstacle/np.linalg.norm(oldtoobstacle)#old to o(bstacle) norm
        #print('oldtoonorm',oldtoonorm)
        obstacleradius=0.05#this is from the customized reacher file!
        relaxcoeff=1.2#1.2 should be the minimum to choose?#1.1#1.1 should be the minimum to choose?#1.5#this might be an important hyperparameter
        oldtoboundary=obstacleradius*oldtoonorm
        #print('oldtoboundary',oldtoboundary)
        otbrelax=relaxcoeff*oldtoboundary#0.07
        reldistold=oldtoobstacle#-oldltoboundary#otbrelax#it should be rel displacement old
        #print('oldposition',oldtoobstacle,'oldtotarget',oldtoobstacle,'oldtoobstacle',oldtoobstacle,'oldvelocity',oldtoobstacle)    
        newposition=observation[0:2]
        newtotarget=observation[2:4]    
        newtoobstacle=observation[4:6]
        newvelocity=observation[6:8]
        #newtoobstacle=observation[2]
        #print('newposition',newposition,'newtotarget',newtotarget,'oldtoobstacle',newtoobstacle,'newvelocity',newvelocity)
        newtoonorm=newtoobstacle/np.linalg.norm(newtoobstacle)
        newtoboundary=obstacleradius*newtoonorm
        ntbrelax=relaxcoeff*newtoboundary#0.07
        reldistnew=newtoobstacle#-newltoboundary#otbrelax#it should be rel displacement old   
        hvalueold = np.linalg.norm(reldistold) ** 2 - np.linalg.norm(otbrelax) ** 2#np.linalg.norm(reldistold) ** 2 - 15 ** 2#get the value of the h function
        #print('oldtoonorm',oldtoonorm,'oldtoboundary',oldtoboundary,'otbrelax',otbrelax,'hvalueold',hvalueold)
        hvaluenew = np.linalg.norm(reldistnew) ** 2 - np.linalg.norm(ntbrelax) ** 2#np.linalg.norm(reldistnew) ** 2 - 15 ** 2#
        hvd=hvaluenew-hvalueold#hvd for h value difference
        #print('newtoonorm',newtoonorm,'newtoboundary',newtoboundary,'ntbrelax',ntbrelax,'hvaluenew',hvaluenew,'hvd',hvd)

        #reldistold=np.zeros(2)
        #reldistnew=np.zeros(2)
        #hvalueold=0
        #hvaluenew=0
        #hvd=0
        extra = {'internal_state': self._env.physics.get_state().copy(),#it seems that you can freely add key-value pairs in extra!
                'discount': time_step.discount,#not show up in spb!
                'constraint': constraint,
                'constraint_cost': constraint_cost,
                'reward': reward,
                'state': old_observation,#it containts the 2 dimensional vector to the center of the obstacle as its 3rd/2nd row!
                'next_state': observation,
                'action': action,##
                "rdo":reldistold,#rdo for relative distance old#array now!
                "rdn": reldistnew,#rdn for relative distance new#array now!
                "hvo": hvalueold,#hvo for h value old#corresponding to the old state
                "hvn":hvaluenew,#hvn for h value new#corresponding to the new state
                "hvd":hvd #hvd for h value difference
                }
        return obs, reward, done, extra#that's how it originally works!
    '''
    def stepsafety2(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = -1 * self._frame_skip#print('running this customized version!')
        print('enter this stepsafety 2 in trajectory generation!')#yes, it enters this step as desired!
        old_observation = _flatten_obs(self._env._task.get_observation(self._env.physics))#what does it contain?#4 rows, 2 columns
        old_constraint = self._env._task.get_constraint(self._env.physics)#I assume the constraint is got from the customized reacher.py
        if self._frozen:
            action = np.zeros_like(action)

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)#propogate
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        observation = _flatten_obs(self._env._task.get_observation(self._env.physics))
        constraint = self._env._task.get_constraint(self._env.physics)
        if constraint:
            self._frozen = True
        # reward -= constraint * 100
        self._n_steps += 1
        # print(self._n_steps)
        done = self._n_steps >= 100

        # TODO: [redacted name :)] update this because these envs are force controlled
        if constraint and not old_constraint:
            constraint_cost = np.linalg.norm(action)
        else:
            constraint_cost = 0

        extra = {'internal_state': self._env.physics.get_state().copy(),#it seems that you can freely add key-value pairs in extra!
                 'discount': time_step.discount,#not show up in spb!
                 'constraint': constraint,
                 'constraint_cost': constraint_cost,
                 'reward': reward,
                 'state': old_observation,#it containts the 2 dimensional vector to the center of the obstacle as its 3rd/2nd row!
                 'next_state': observation,
                 'action': action}

        return obs, reward, done, extra#that's how it originally works!
    
    def reset(self, **kwargs):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        self._frozen = False
        self._n_steps = 0
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
