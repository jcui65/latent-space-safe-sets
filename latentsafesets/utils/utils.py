
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')

import torch
import torch.nn as nn
import numpy as np

import logging
import os
import json
from datetime import datetime
import random
from tqdm import tqdm, trange

from latentsafesets.utils.replay_buffer_encoded import EncodedReplayBuffer
from latentsafesets.utils.replay_buffer import ReplayBuffer
from gym.wrappers import FrameStack

log = logging.getLogger("utils")


files = {
    'spb': [
        'SimplePointBot', 'SimplePointBotConstraints'
    ],
    'apb': [
        'AccelerationPointBot', 'AccelerationPointBotConstraint'
    ],
    'reacher': [
        'Reacher', 'ReacherConstraints', 'ReacherInteractions'
    ]
}


def seed(seed):
    # torch.set_deterministic(True)
    if seed == -1:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_file_prefix(exper_name=None, seed=-1):
    if exper_name is not None:
        folder = os.path.join('outputs', exper_name)
    else:
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d/%H-%M-%S")#year month day/hour minute second
        folder = os.path.join('outputs', date_string)#outputs/year-month-day/hour-minute-second
    if seed != -1:#that folder ended in 0001!!!
        folder = os.path.join(folder, str(seed))
    return folder#outputs/year-month-day/hour-minute-second


def init_logging(folder, file_level=logging.INFO, console_level=logging.DEBUG):
    # set up logging to file
    logging.basicConfig(level=file_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=os.path.join(folder, 'log.txt'),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(console_level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def save_trajectories(trajectories, file):
    if not os.path.exists(file):
        os.makedirs(file)
    else:
        raise RuntimeError("Directory %s already exists." % file)

    for i, traj in enumerate(trajectories):
        save_trajectory(traj, file, i)


def save_trajectory(trajectory, file, n):#file: data/SimplePointBot or data/SimplePointBotConstraints
    im_fields = ('obs', 'next_obs')
    for field in im_fields:#obs, next_obs, .json do their jobs, respectively
        if field in trajectory[0]:#a dictionary, trajectory [0] is the 0th/first step/frame
            dat = np.array([frame[field] for frame in trajectory], dtype=np.uint8)#
            #it is 100 pieces of 3-channel image of obs or next_obs
            np.save(os.path.join(file, "%d_%s.npy" % (n, field)), dat)#save the images in .npy file
    traj_no_ims = [{key: frame[key] for key in frame if key not in im_fields}
                   for frame in trajectory]#trajectory contains 100 frames
    with open(os.path.join(file, "%d.json" % n), "w") as f:
        json.dump(traj_no_ims, f)#separate trajectory info from images


def load_trajectories(num_traj, file):#data/simplepointbot
    log.info('Loading trajectories from %s' % file)#data/SimplePointBot

    if not os.path.exists(file):
        raise RuntimeError("Could not find directory %s." % file)
    trajectories = []
    iterator = range(num_traj) if num_traj <= 200 else trange(num_traj)
    for i in iterator:#50
        if not os.path.exists(os.path.join(file, '%d.json' % i)):#e.g. 0.json
            log.info('Could not find %d' % i)
            continue
        im_fields = ('obs', 'next_obs')
        with open(os.path.join(file, '%d.json' % i), 'r') as f:#read the json file!
            trajectory = json.load(f)#1 piece traj info without 2 images 100 time steps
        im_dat = {}#image_data

        for field in im_fields:
            f = os.path.join(file, "%d_%s.npy" % (i, field))#obs and next_obs
            if os.path.exists(file):
                dat = np.load(f)
                im_dat[field] = dat.astype(np.uint8)#100 images of obs and next_obs

        for j, frame in list(enumerate(trajectory)):#each frame in one trajectory
            for key in im_dat:#from obs and next_obs
                frame[key] = im_dat[key][j]#the frame is the jth frame in 1 traj
        trajectories.append(trajectory)#now you recover the full trajectory info with images

    return trajectories#that is a sequence/buffer/pool of trajs including images


def load_replay_buffer(params, encoder=None, first_only=False):#it doesn't have traj parameter!
    log.info('Loading data')
    trajectories = []#SimplePointBot or SimplePointBotConstraints
    for directory, num in list(zip(params['data_dirs'], params['data_counts'])):#safe & obstacle
        real_dir = os.path.join('/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets','data', directory)#get the trajectories
        trajectories += load_trajectories(num, file=real_dir)#now you have 50+50=100 pieces of trajs each containing 100 time steps
        if first_only:
            print('wahoo')
            break

    log.info('Populating replay buffer')#find correspondence in the cmd output

    # Shuffle array so that when the replay fills up it doesn't remove one dataset before the other
    random.shuffle(trajectories)
    if encoder is not None:#replay buffer finally comes in!
        replay_buffer = EncodedReplayBuffer(encoder, params['buffer_size'])#35000 for spb
    else:
        replay_buffer = ReplayBuffer(params['buffer_size'])

    for trajectory in tqdm(trajectories):#trajectory is 1 traj having 100 steps
        replay_buffer.store_transitions(trajectory)#22
    #finally, the self.data, a dict in the replay_buffer is filled with values from 100 trajs, each containing 100 steps
    return replay_buffer
    #each key in self.data, its value is a numpy array containing 10000=100*100 pieces of info/data of each transition

def make_env(params, monitoring=False):
    from latentsafesets.envs import SimplePointBot, PushEnv, SimpleVideoSaver
    env_name = params['env']
    if env_name == 'spb':
        env = SimplePointBot(True)#the same environment, different teacher!
    elif env_name == 'reacher':
        import dmc2gym

        env = dmc2gym.make(domain_name='reacher', task_name='hard', seed=params['seed'],
                           from_pixels=True, visualize_reward=False, channels_first=True)
    elif env_name == 'push':
        env = PushEnv()
    else:
        raise NotImplementedError

    if params['frame_stack'] > 1:#connecting subsequent/consequtive images
        env = FrameStack(env, params['frame_stack'])

    if monitoring:
        env = SimpleVideoSaver(env, os.path.join(params['logdir'], 'videos'))

    return env


def make_modules(params, ss=False, val=False, dyn=False,
                 gi=False, constr=False):
    from latentsafesets.modules import VanillaVAE, ValueEnsemble, \
        ValueFunction, PETSDynamics, GoalIndicator, ConstraintEstimator, BCSafeSet, \
        BellmanSafeSet
    import latentsafesets.utils.pytorch_utils as ptu

    modules = {}

    encoder = VanillaVAE(params)#initialize/instantiate the VAE
    if params['enc_checkpoint']:
        encoder.load(params['enc_checkpoint'])#load the parameters of the VAE at specfic checkpoints!
    modules['enc'] = encoder

    if ss:
        safe_set_type = params['safe_set_type']
        if safe_set_type == 'bc':
            safe_set = BCSafeSet(encoder, params)#initialize/instantiate the safe set
        elif safe_set_type == 'bellman':
            safe_set = BellmanSafeSet(encoder, params)
        else:
            raise NotImplementedError
        if params['safe_set_checkpoint']:#if we are gonna train it separately!
            safe_set.load(params['safe_set_checkpoint'])
        modules['ss'] = safe_set

    if val:
        if params['val_ensemble']:#what are the difference
            value_func = ValueEnsemble(encoder, params).to(ptu.TORCH_DEVICE)
        else:
            value_func = ValueFunction(encoder, params).to(ptu.TORCH_DEVICE)
        if params['val_checkpoint']:
            value_func.load(params['val_checkpoint'])
        modules['val'] = value_func

    if dyn:
        dynamics = PETSDynamics(encoder, params)
        if params['dyn_checkpoint']:
            dynamics.load(params['dyn_checkpoint'])
        modules['dyn'] = dynamics

    if gi:
        goal_indicator = GoalIndicator(encoder, params).to(ptu.TORCH_DEVICE)
        if params['gi_checkpoint']:
            goal_indicator.load(params['gi_checkpoint'])
        modules['gi'] = goal_indicator

    if constr:
        constraint = ConstraintEstimator(encoder, params).to(ptu.TORCH_DEVICE)
        if params['constr_checkpoint']:
            constraint.load(params['constr_checkpoint'])
        modules['constr'] = constraint

    return modules


def make_modulessafety(params, ss=False, val=False, dyn=False,
                 gi=False, constr=False,cbfd=False):
    from latentsafesets.modules import VanillaVAE, ValueEnsemble, \
        ValueFunction, PETSDynamics, GoalIndicator, ConstraintEstimator, BCSafeSet, \
        BellmanSafeSet, CBFdotEstimator
    import latentsafesets.utils.pytorch_utils as ptu

    modules = {}

    encoder = VanillaVAE(params)#initialize/instantiate the VAE
    if params['enc_checkpoint']:
        encoder.load(params['enc_checkpoint'])#load the parameters of the VAE at specfic checkpoints!
    modules['enc'] = encoder

    if ss:
        safe_set_type = params['safe_set_type']
        if safe_set_type == 'bc':
            safe_set = BCSafeSet(encoder, params)#initialize/instantiate the safe set
        elif safe_set_type == 'bellman':
            safe_set = BellmanSafeSet(encoder, params)
        else:
            raise NotImplementedError
        if params['safe_set_checkpoint']:#if we are gonna train it separately!
            safe_set.load(params['safe_set_checkpoint'])
        modules['ss'] = safe_set

    if val:
        if params['val_ensemble']:#what are the difference
            value_func = ValueEnsemble(encoder, params).to(ptu.TORCH_DEVICE)
        else:
            value_func = ValueFunction(encoder, params).to(ptu.TORCH_DEVICE)
        if params['val_checkpoint']:
            value_func.load(params['val_checkpoint'])
        modules['val'] = value_func

    if dyn:
        dynamics = PETSDynamics(encoder, params)
        if params['dyn_checkpoint']:
            dynamics.load(params['dyn_checkpoint'])
        modules['dyn'] = dynamics

    if gi:
        goal_indicator = GoalIndicator(encoder, params).to(ptu.TORCH_DEVICE)
        if params['gi_checkpoint']:
            goal_indicator.load(params['gi_checkpoint'])
        modules['gi'] = goal_indicator

    if constr:
        constraint = ConstraintEstimator(encoder, params).to(ptu.TORCH_DEVICE)
        if params['constr_checkpoint']:
            constraint.load(params['constr_checkpoint'])
        modules['constr'] = constraint

    if cbfd:
        cbfdot = CBFdotEstimator(encoder, params).to(ptu.TORCH_DEVICE)
        if params['cbfd_checkpoint']:
            cbfdot.load(params['cbfd_checkpoint'])
        modules['cbfd'] = cbfdot
        print(modules['cbfd'])
    return modules


class RunningMeanStd(nn.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        super(RunningMeanStd, self).__init__()

        from latentsafesets.utils.pytorch_utils import TORCH_DEVICE

        # We store these as parameters so they'll be stored in dynamic model state dicts
        self.mean = nn.Parameter(torch.zeros(shape, dtype=torch.float32, device=TORCH_DEVICE),
                                 requires_grad=False)
        self.var = nn.Parameter(torch.ones(shape, dtype=torch.float32, device=TORCH_DEVICE),
                                requires_grad=False)
        self.count = nn.Parameter(torch.tensor(epsilon))

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)#see 248

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = nn.Parameter(new_mean, requires_grad=False)
        self.var = nn.Parameter(new_var, requires_grad=False)
        self.count = nn.Parameter(new_count, requires_grad=False)

