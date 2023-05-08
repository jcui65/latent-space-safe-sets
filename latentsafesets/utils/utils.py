
import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')
sys.path.insert(1, '/home/cuijin/Project6remote/latent-space-safe-sets')
import torch
import torch.nn as nn
import numpy as np

import logging
import os
import json
from datetime import datetime
import random
from tqdm import tqdm, trange

from latentsafesets.utils.replay_buffer_encoded import EncodedReplayBuffer, EncodedReplayBuffer_expensive2
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
    #if seed != -1:#that folder ended in 0001!!!#I change/delete it!
        #folder = os.path.join(folder, str(seed))
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

    for i, traj in enumerate(trajectories):#go through all the trajectories!!!
        save_trajectory(traj, file, i)#save each trajectory


def save_trajectory(trajectory, file, n):#file: data/SimplePointBot or data/SimplePointBotConstraints
    im_fields = ('obs', 'next_obs')
    for field in im_fields:#obs, next_obs, .json do their jobs, respectively
        if field in trajectory[0]:#a dictionary, trajectory [0] is the 0th/first step/frame
            tr0=trajectory[0]
            #print('trajectory',tr0)
            #print('tr0[field]',tr0[field])
            dat = np.array([frame[field] for frame in trajectory], dtype=np.uint8)#
            #it is 100 pieces of 3-channel image of obs or next_obs
            np.save(os.path.join(file, "%d_%s.npy" % (n, field)), dat)#save the images in .npy file
    traj_no_ims = [{key: frame[key] for key in frame if key not in im_fields}
                   for frame in trajectory]#trajectory contains 100 frames
    with open(os.path.join(file, "%d.json" % n), "w") as f:
        json.dump(traj_no_ims, f)#separate trajectory info from images

def save_trajectory_relative(trajectory, file, n):#file: data/SimplePointBot or data/SimplePointBotConstraints
    im_fields = ('obs', 'next_obs','obs_relative', 'next_obs_relative')
    for field in im_fields:#obs, next_obs, .json do their jobs, respectively
        if field in trajectory[0]:#a dictionary, trajectory [0] is the 0th/first step/frame
            dat = np.array([frame[field] for frame in trajectory], dtype=np.uint8)#
            #print('dat.shape',dat.shape)#(100, 3, 64, 64)
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
    iterator = range(num_traj) if num_traj <= 200 else trange(num_traj)#maybe a bug source?
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

def load_trajectories_relative(num_traj, file):#data/simplepointbot#num_traj is the num of traj in each folder!
    log.info('Loading trajectories from %s' % file)#data/SimplePointBot

    if not os.path.exists(file):
        raise RuntimeError("Could not find directory %s." % file)
    trajectories = []#num_traj is the data_count!
    iterator = range(num_traj) if num_traj <= 200 else trange(num_traj)
    for i in iterator:#50
        if not os.path.exists(os.path.join(file, '%d.json' % i)):#e.g. 0.json
            log.info('Could not find %d' % i)
            continue
        im_fields = ('obs', 'next_obs','obs_relative', 'next_obs_relative')#('obs', 'next_obs')
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
                #print('key',key)#(3,64,64)
                #print('frame[key].shape',frame[key].shape)
        trajectories.append(trajectory)#now you recover the full trajectory info with images

    return trajectories#that is a sequence/buffer/pool of trajs including images

def load_replay_buffer(params, encoder=None, first_only=False):#it doesn't have traj parameter!
    log.info('Loading data')
    trajectories = []#SimplePointBot or SimplePointBotConstraints
    for directory, num in list(zip(params['data_dirs'], params['data_counts'])):#safe 50 & obstacle 50
        #real_dir = os.path.join('/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets','data', directory)#get the trajectories
        if params['light']=='ls3':
            if params['datasetnumber']==1:
                real_dir = os.path.join('', 'datals3',directory)  #old data!#',directory)  #new data!#
            elif params['datasetnumber']==2 or params['datasetnumber']==3:#only load the data, not involving data_images!
                real_dir = os.path.join('', 'data',directory)  #new data!#ls3',directory)  #old data!#
        else:
            real_dir = os.path.join('', 'data',directory)  #
        trajectories += load_trajectories(num, file=real_dir)#now you have 50+50=100 pieces of trajs each containing 100 time steps
        if first_only:
            print('wahoo')
            break

    log.info('Populating replay buffer')#find correspondence in the cmd output

    # Shuffle array so that when the replay fills up it doesn't remove one dataset before the other
    random.shuffle(trajectories)
    if encoder is not None:#replay buffer finally comes in!
        replay_buffer = EncodedReplayBuffer(encoder, params['buffer_size'],params['mean'])#35000 for spb, 25000 for reacher
        #print('load encoded buffer!')#load encoded buffer!
    else:
        replay_buffer = ReplayBuffer(params['buffer_size'])
        #print('load plain buffer!')
    for trajectory in tqdm(trajectories):#trajectory is 1 traj having 100 steps, trajectories is a list of many trajectorys!
        replay_buffer.store_transitions(trajectory)#22#
    #finally, the self.data, a dict in the replay_buffer is filled with values from 100 trajs, each containing 100 steps
    return replay_buffer
    #each key in self.data, its value is a numpy array containing 10000=100*100 pieces of info/data of each transition

def load_replay_buffer_unsafe(params, encoder=None, first_only=False):#it doesn't have traj parameter!
    log.info('Loading data unsafe demonstration!')
    trajectories = []#SimplePointBot or SimplePointBotConstraints
    for directory, num in list(zip(params['data_dirs'], params['data_counts'])):#safe 50 & obstacle 50
        #real_dir = os.path.join('/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets','data', directory)#get the trajectories
        if directory=='ReacherConstraintdense1' or directory=='ReacherConstraintdense2' or directory=='PushOutbursts2':
            if params['light']=='ls3':
                if params['datasetnumber']==1:
                    real_dir = os.path.join('', 'datals3',directory)  #old data!#',directory)  #new data!#
                elif params['datasetnumber']==2 or params['datasetnumber']==3:
                    real_dir = os.path.join('', 'data',directory)  #new data!#ls3',directory)  #old data!#
            else:
                real_dir = os.path.join('', 'data',directory)  #
            trajectories += load_trajectories(num, file=real_dir)#now you have 50+50=100 pieces of trajs each containing 100 time steps
            if first_only:
                print('wahoo')
                break

    log.info('Populating replay buffer unsafe demonstration!')#find correspondence in the cmd output

    # Shuffle array so that when the replay fills up it doesn't remove one dataset before the other
    random.shuffle(trajectories)
    if encoder is not None:#replay buffer finally comes in!
        replay_buffer = EncodedReplayBuffer(encoder, params['buffer_size'],params['mean'])#35000 for spb, 25000 for reacher!
        print('load encoded buffer!')
    else:
        replay_buffer = ReplayBuffer(params['buffer_size'])
        print('load plain buffer!')

    for trajectory in tqdm(trajectories):#trajectory is 1 traj having 100 steps
        replay_buffer.store_transitions(trajectory)#22
    #finally, the self.data, a dict in the replay_buffer is filled with values from 100 trajs, each containing 100 steps
    return replay_buffer
    #each key in self.data, its value is a numpy array containing 10000=100*100 pieces of info/data of each transition

def load_replay_buffer_relative(params, encoder=None, first_only=False):#it doesn't have traj parameter!
    log.info('Loading data')
    trajectories = []#SimplePointBot or SimplePointBotConstraints
    for directory, num in list(zip(params['data_dirs'], params['data_counts'])):#safe & obstacle
        #real_dir = os.path.join('/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets','data', directory)#get the trajectories
        real_dir = os.path.join('', 'data_relative',
                                directory)  #
        trajectories += load_trajectories_relative(num, file=real_dir)#now you have 50+50=100 pieces of trajs each containing 100 time steps
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

def load_replay_buffer_relative_expensive2(params, encoder=None, encoder2=None, first_only=False):#it doesn't have traj parameter!
    log.info('Loading data')
    trajectories = []#SimplePointBot or SimplePointBotConstraints
    for directory, num in list(zip(params['data_dirs'], params['data_counts'])):#safe & obstacle
        #real_dir = os.path.join('/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets','data', directory)#get the trajectories
        real_dir = os.path.join('', 'data_relative',
                                directory)  #
        trajectories += load_trajectories_relative(num, file=real_dir)#now you have 50+50=100 pieces of trajs each containing 100 time steps
        if first_only:
            print('wahoo')
            break

    log.info('Populating replay buffer')#find correspondence in the cmd output

    # Shuffle array so that when the replay fills up it doesn't remove one dataset before the other
    random.shuffle(trajectories)
    if encoder is not None:#replay buffer finally comes in!
        #replay_buffer = EncodedReplayBuffer(encoder, params['buffer_size'])#35000 for spb
        replay_buffer = EncodedReplayBuffer_expensive2(encoder,encoder2, params['buffer_size'])#35000 for spb
        #replay_buffer = EncodedReplayBuffer(encoder, params['buffer_size'])  # 35000 for spb
    else:
        replay_buffer = ReplayBuffer(params['buffer_size'])

    for trajectory in tqdm(trajectories):#trajectory is 1 traj having 100 steps
        replay_buffer.store_transitions(trajectory)#22
    #finally, the self.data, a dict in the replay_buffer is filled with values from 100 trajs, each containing 100 steps
    return replay_buffer

def make_env(params, monitoring=False):
    from latentsafesets.envs import SimplePointBot, PushEnv, SimpleVideoSaver
    env_name = params['env']
    if env_name == 'spb':
        env = SimplePointBot(True)#the same environment, different teacher!
    elif env_name == 'reacher':
        import dmc2gym

        env = dmc2gym.make(domain_name='reacher', task_name='hard', seed=params['seed'],#it is choosing the hard option
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
        BellmanSafeSet, CBFdotEstimatorlatentplana #CBFdotEstimatorlatent
    import latentsafesets.utils.pytorch_utils as ptu

    modules = {}
    
    env=params['env']
    dsn=params['datasetnumber']
    vsn=params['vaesnumber']
    if env=='reacher':
        if dsn==1:
            if vsn=='d1v1':
                params['enc_checkpoint']='outputs/2023-02-18/19-47-57/1/vae.pth'#old reacher#
                params['safe_set_checkpoint']='outputs/2023-02-19/16-20-28/1/initial_train/ss.pth'#old reacher#
                params['val_checkpoint']='outputs/2023-02-19/16-20-28/1/initial_train/val.pth'#old reacher#
                params['dyn_checkpoint']='outputs/2023-02-19/16-20-28/1/initial_train/dyn.pth'#old reacher#
                params['gi_checkpoint']='outputs/2023-02-19/16-20-28/1/initial_train/gi.pth'#old reacher#
                params['constr_checkpoint']='outputs/2023-02-19/16-20-28/1/initial_train/constr.pth'#old reacher#
                params['cbfd_checkpoint']=None
            elif vsn=='d1v2':
                params['enc_checkpoint']='outputs/2023-04-08/01-57-17/vae.pth'#old 2 reacher#
                params['safe_set_checkpoint']='outputs/2023-04-08/12-39-06/1/initial_train/ss.pth'#old 2 reacher#
                params['val_checkpoint']='outputs/2023-04-08/12-39-06/1/initial_train/val.pth'#old 2 reacher#
                params['dyn_checkpoint']='outputs/2023-04-08/12-39-06/1/initial_train/dyn.pth'#old 2 reacher#
                params['gi_checkpoint']='outputs/2023-04-08/12-39-06/1/initial_train/gi.pth'#old 2 reacher#
                params['constr_checkpoint']='outputs/2023-04-08/12-39-06/1/initial_train/constr.pth'#old 2 reacher#
                params['cbfd_checkpoint']=None
            elif vsn=='no1':
                params['enc_checkpoint']='outputs/2023-02-18/19-47-57/1/vae.pth'#old reacher#
                params['safe_set_checkpoint']=None
                params['val_checkpoint']=None
                params['dyn_checkpoint']=None
                params['gi_checkpoint']=None
                params['constr_checkpoint']=None
                params['cbfd_checkpoint']=None
            elif vsn=='no2':
                params['enc_checkpoint']='outputs/2023-04-08/01-57-17/vae.pth'#old 2 reacher#
                params['safe_set_checkpoint']=None
                params['val_checkpoint']=None
                params['dyn_checkpoint']=None
                params['gi_checkpoint']=None
                params['constr_checkpoint']=None
                params['cbfd_checkpoint']=None
        if dsn==2:
            if vsn=='d2v1':
                params['enc_checkpoint']='outputs/2023-04-05/19-29-19/vae.pth'#new reacher#
                if params['mean']=='sample':
                    params['safe_set_checkpoint']='outputs/2023-04-07/19-43-49/1/initial_train/ss.pth'#really new reacher#
                    params['val_checkpoint']='outputs/2023-04-07/19-43-49/1/initial_train/val.pth'#really new reacher
                    params['dyn_checkpoint']='outputs/2023-04-07/19-43-49/1/initial_train/dyn.pth'#really new reacher
                    params['gi_checkpoint']='outputs/2023-04-07/19-43-49/1/initial_train/gi.pth'#really new reacher
                    params['constr_checkpoint']='outputs/2023-04-07/19-43-49/1/initial_train/constr.pth'#really new reacher
                    params['cbfd_checkpoint']='outputs/2023-04-09/01-55-20/1/initial_train/cbfd.pth'#really new reacher#'outputs/2023-04-23/17-51-53/cbfd_10000.pth'#10000 epochs again#'outputs/2023-04-23/17-51-53/cbfd.pth'#50000 epochs again#None#'outputs/2023-04-23/13-21-39/cbfd.pth'#50000 epochs of pretraining#really new reacher#
                elif params['mean']=='mean':
                    params['safe_set_checkpoint']='outputs/2023-04-26/00-04-53/1/initial_train/ss.pth'#mean reacher#
                    params['val_checkpoint']='outputs/2023-04-26/00-04-53/1/initial_train/val.pth'#mean reacher
                    params['dyn_checkpoint']='outputs/2023-04-26/00-04-53/1/initial_train/dyn.pth'#mean reacher
                    params['gi_checkpoint']='outputs/2023-04-26/00-04-53/1/initial_train/gi.pth'#mean reacher
                    params['constr_checkpoint']='outputs/2023-04-26/00-04-53/1/initial_train/constr.pth'#mean reacher
                    params['cbfd_checkpoint']='outputs/2023-04-26/00-04-53/1/initial_train/cbfd_10000.pth'#mean reacher
            elif vsn=='d2v2':
                params['enc_checkpoint']='outputs/2023-04-08/02-02-17/vae.pth'#new 2 reacher#
                params['safe_set_checkpoint']='outputs/2023-04-08/12-42-19/1/initial_train/ss.pth'#new 2 reacher#
                params['val_checkpoint']='outputs/2023-04-08/12-42-19/1/initial_train/val.pth'#new 2 reacher#
                params['dyn_checkpoint']='outputs/2023-04-08/12-42-19/1/initial_train/dyn.pth'#new 2 reacher#
                params['gi_checkpoint']='outputs/2023-04-08/12-42-19/1/initial_train/gi.pth'#new 2 reacher#
                params['constr_checkpoint']='outputs/2023-04-08/12-42-19/1/initial_train/constr.pth'#new 2 reacher#
                params['cbfd_checkpoint']=None
            elif vsn=='no1':
                params['enc_checkpoint']='outputs/2023-04-05/19-29-19/vae.pth'#new reacher#
                params['safe_set_checkpoint']=None
                params['val_checkpoint']=None
                params['dyn_checkpoint']=None
                params['gi_checkpoint']=None
                params['constr_checkpoint']=None
                params['cbfd_checkpoint']=None
            elif vsn=='no2':
                params['enc_checkpoint']='outputs/2023-04-08/02-02-17/vae.pth'#new 2 reacher#
                params['safe_set_checkpoint']=None
                params['val_checkpoint']=None
                params['dyn_checkpoint']=None
                params['gi_checkpoint']=None
                params['constr_checkpoint']=None
                params['cbfd_checkpoint']=None
    '''
    elif env=='push':
        if dsn==1:
            if vsn=='d1v1':
                params['enc_checkpoint']=
                params['safe_set_checkpoint']=
                params['val_checkpoint']=
                params['dyn_checkpoint']=
                params['gi_checkpoint']=
                params['constr_checkpoint']=
                params['cbfd_checkpoint']=None
            elif vsn=='d1v2':
                params['enc_checkpoint']=
                params['safe_set_checkpoint']=
                params['val_checkpoint']=
                params['dyn_checkpoint']=
                params['gi_checkpoint']=
                params['constr_checkpoint']=
                params['cbfd_checkpoint']=None
            elif vsn=='no1':
                params['enc_checkpoint']=
                params['safe_set_checkpoint']=
                params['val_checkpoint']=
                params['dyn_checkpoint']=
                params['gi_checkpoint']=
                params['constr_checkpoint']=
                params['cbfd_checkpoint']=
            elif vsn=='no2':
                params['enc_checkpoint']=
                params['safe_set_checkpoint']=
                params['val_checkpoint']=
                params['dyn_checkpoint']=
                params['gi_checkpoint']=
                params['constr_checkpoint']=
                params['cbfd_checkpoint']=
        if dsn==2:
            if vsn=='d2v1':
                params['enc_checkpoint']=
                params['safe_set_checkpoint']=
                params['val_checkpoint']=
                params['dyn_checkpoint']=
                params['gi_checkpoint']=
                params['constr_checkpoint']=
                params['cbfd_checkpoint']=None
            elif vsn=='d2v2':
                params['enc_checkpoint']=
                params['safe_set_checkpoint']=
                params['val_checkpoint']=
                params['dyn_checkpoint']=
                params['gi_checkpoint']=
                params['constr_checkpoint']=
                params['cbfd_checkpoint']=None
            elif vsn=='no1':
                params['enc_checkpoint']=
                params['safe_set_checkpoint']=
                params['val_checkpoint']=
                params['dyn_checkpoint']=
                params['gi_checkpoint']=
                params['constr_checkpoint']=
                params['cbfd_checkpoint']=
            elif vsn=='no2':
                params['enc_checkpoint']=
                params['safe_set_checkpoint']=
                params['val_checkpoint']=
                params['dyn_checkpoint']=
                params['gi_checkpoint']=
                params['constr_checkpoint']=
                params['cbfd_checkpoint']='''
    encoder = VanillaVAE(params)#initialize/instantiate the VAE
    if params['enc_checkpoint']:
        encoder.load(params['enc_checkpoint'])#load the parameters of the VAE at specfic checkpoints!
        print('params[enc_checkpoint]',params['enc_checkpoint'])#it is working as expected!
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
            print('params[safe_set_checkpoint]',params['safe_set_checkpoint'])#it is working as expected!
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
        #cbfdot = CBFdotEstimatorlatent(encoder, params).to(ptu.TORCH_DEVICE)
        cbfdot = CBFdotEstimatorlatentplana(encoder, params).to(ptu.TORCH_DEVICE)
        if params['cbfd_checkpoint']:
            print('params[cbfd_checkpoint]',params['cbfd_checkpoint'])#it is working as expected!
            cbfdot.load(params['cbfd_checkpoint'])
        modules['cbfd'] = cbfdot
        #print(modules['cbfd'])
    return modules
'''
def make_modulessafetylight(params, val=False, dyn=False,
                 gi=False, cbfd=False):
    from latentsafesets.modules import VanillaVAE, ValueEnsemble, \
        ValueFunction, PETSDynamics, GoalIndicator, CBFdotEstimatorlatentplana #CBFdotEstimatorlatent
    import latentsafesets.utils.pytorch_utils as ptu

    modules = {}

    encoder = VanillaVAE(params)#initialize/instantiate the VAE
    if params['enc_checkpoint']:
        encoder.load(params['enc_checkpoint'])#load the parameters of the VAE at specfic checkpoints!
    modules['enc'] = encoder

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

    if cbfd:
        #cbfdot = CBFdotEstimatorlatent(encoder, params).to(ptu.TORCH_DEVICE)
        cbfdot = CBFdotEstimatorlatentplana(encoder, params).to(ptu.TORCH_DEVICE)
        if params['cbfd_checkpoint']:
            cbfdot.load(params['cbfd_checkpoint'])
        modules['cbfd'] = cbfdot
        #print(modules['cbfd'])
    return modules
'''
def make_modulessafetyexpensive(params, ss=False, val=False, dyn=False,
                 gi=False, constr=False,cbfd=False):
    from latentsafesets.modules import VanillaVAE, ValueEnsemble, \
        ValueFunction, PETSDynamics, GoalIndicator, ConstraintEstimator, BCSafeSet, \
        BellmanSafeSet, CBFdotEstimatorlatentplana #CBFdotEstimatorlatent
    import latentsafesets.utils.pytorch_utils as ptu

    modules = {}

    encoder = VanillaVAE(params)#initialize/instantiate the VAE
    encoder2 = VanillaVAE(params)  # initialize/instantiate the VAE
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

    if params['enc_checkpoint2']:
        encoder2.load(params['enc_checkpoint2'])#load the parameters of the VAE at specfic checkpoints!
    modules['enc2'] = encoder2

    if cbfd:
        #cbfdot = CBFdotEstimatorlatent(encoder, params).to(ptu.TORCH_DEVICE)
        cbfdot = CBFdotEstimatorlatentplana(encoder2, params).to(ptu.TORCH_DEVICE)
        if params['cbfd_checkpoint']:
            cbfdot.load(params['cbfd_checkpoint'])
        modules['cbfd'] = cbfdot
        #print(modules['cbfd'])

    return modules

def make_modulessafetyexpensive2(params, ss=False, val=False, dyn=False,
                 gi=False, constr=False,cbfd=False, dyn2=False):
    from latentsafesets.modules import VanillaVAE, ValueEnsemble, \
        ValueFunction, PETSDynamics, GoalIndicator, ConstraintEstimator, BCSafeSet, \
        BellmanSafeSet, CBFdotEstimatorlatentplana, VanillaVAE2, PETSDynamics2#CBFdotEstimatorlatent
    import latentsafesets.utils.pytorch_utils as ptu

    modules = {}

    encoder = VanillaVAE(params)#initialize/instantiate the VAE
    encoder2 = VanillaVAE2(params)  # initialize/instantiate the VAE
    if params['enc_checkpoint']:
        encoder.load(params['enc_checkpoint'])#load the parameters of the VAE at specfic checkpoints!
        print(params['enc_checkpoint'])
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

    if params['enc_checkpoint2']:
        print(params['enc_checkpoint2'])
        encoder2.load(params['enc_checkpoint2'])#load the parameters of the VAE at specfic checkpoints!
    modules['enc2'] = encoder2

    if cbfd:
        #cbfdot = CBFdotEstimatorlatent(encoder, params).to(ptu.TORCH_DEVICE)
        cbfdot = CBFdotEstimatorlatentplana(encoder2, params).to(ptu.TORCH_DEVICE)
        if params['cbfd_checkpoint']:
            print(params['cbfd_checkpoint'])
            cbfdot.load(params['cbfd_checkpoint'])
        modules['cbfd'] = cbfdot
        #print(modules['cbfd'])

    if dyn2:
        dynamics2 = PETSDynamics2(encoder2, params)#verified
        if params['dyn_checkpoint2']:
            print(params['dyn_checkpoint2'])
            dynamics2.load(params['dyn_checkpoint2'])
        modules['dyn2'] = dynamics2

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

