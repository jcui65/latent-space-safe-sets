
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/cuijin/Project6remote/latent-space-safe-sets')

from latentsafesets.policy import CEMSafeSetPolicy#this is the class!
import latentsafesets.utils as utils
import latentsafesets.utils.plot_utils as pu
#from latentsafesets.utils.arg_parser import parse_args
from latentsafesets.utils.arg_parser_push import parse_args
from latentsafesets.rl_trainers import MPCTrainer
import latentsafesets.utils.pytorch_utils as ptu
import torch
from torch.autograd.functional import jacobian
import os
import logging
from tqdm import trange#mainly for showing the progress bar
import numpy as np
import pprint

from latentsafesets.modules import VanillaVAE, CBFdotEstimator,CBFdotEstimatorlatentplana
from datetime import datetime
#provides a capability to “pretty-print” arbitrary Python data structures in a form that can be used as input to the interpreter
#log = logging.getLogger("main")#some logging stuff


if __name__ == '__main__':
    params = parse_args()#get the parameters from parse_args, see arg_parser.py
    # Misc preliminaries
    repeattimes=params['repeat_times']
    initdhz=params['dhz']
    for m in range(repeattimes):
        num_updates = params['num_updates']#default 25 in spb reacher/100 in push
        traj_per_update = params['traj_per_update']#default 10
        slopexy=slopeyz=slopezh=slopeyh=slopexh=np.zeros((num_updates*params['horizon']))
        slopexys=slopeyzs=slopezhs=slopeyhs=slopexhs=np.zeros((num_updates*params['horizon']))
        slopezq=slopeyq=slopexq=qzuno=np.zeros((num_updates*params['horizon']))
        slopezqs=slopeyqs=slopexqs=qzunos=pdnarray=np.zeros((num_updates*params['horizon']))
        slopexyu=slopeyzu=slopezhu=slopeyhu=slopexhu=np.zeros((num_updates*params['horizon']))
        piece=0#which piece of trajectory? this piece
        eps=1e-10
        lipxy=lipyz=lipzh=lipyh=lipxh=lipzq=lipyq=lipxq=0
        lipxysafe=lipyzsafe=lipzhsafe=lipyhsafe=lipxhsafe=lipzqsafe=lipyqsafe=lipxqsafe=0
        lipxyunsafe=lipyzunsafe=lipzhunsafe=lipyhunsafe=lipxhunsafe=0
        gammadyn=gammadyns=100#start from a very big number!
        pdnsafe=pdn=0
        params['dhz']=initdhz#(1-cbfalpha)*dhzoriginal+cbfalpha*episodiccbfdhz
        #params['seed']=23
        log = logging.getLogger("main")#some logging stuffs
        seed=params['seed']
        #print('seed',seed)#works as expected!
        utils.seed(params['seed'])#around line 10, the default is -1, meaning random seed
        #folder = os.path.join(folder, str(seed))
        #logdir = params['logdir']#around line 35
        logdirbeforeseed = params['logdir']#around line 35
        logdir=os.path.join(logdirbeforeseed, str(seed))
        os.makedirs(logdir)#e.g.: 'outputs/2022-07-15/17-41-16'
        #logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M:%S',filename=os.path.join(logdir, 'logjianning.txt'),filemode='w')
        #utils.init_loggingcjn(logdir)#record started!#only recording the current parameters!
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d-%H-%M-%S")#year month day/hour minute second
        f = open(logdir+"/logjianning"+date_string+".txt", "a")#so that I can write my own comments even when the simulation is still running!!!
        f.write('The alpha: %1.3f\t'%(params['cbfdot_thresh']))
        f.write('H: %d\t'%(int(params['plan_hor'])))
        f.write('d_{thres}=0.2\t')
        f.write('action_type: %s\t'%(params['action_type']))
        f.write('conservativeness: %s, reward_type: %s, lightness: %s\t'%(params['conservative'],params['reward_type'],params['light']))
        f.write('push_cbf_strategy: %d\n'%(params['push_cbf_strategy']))
        f.write('reduce_horizon: %s, 0 or 1: %s\n'%(params['reduce_horizon'],params['zero_one']))
        f.write('mean: %s, dhz: %f, dhdmax: %f\n'%(params['mean'],params['dhz'],params['dhdmax']))
        f.write('nosigma: %f, nosigmadhz: %f, dynamic_dhz: %s, idea: %s, unsafe_buffer:%s, reg_lipschitz:%s\n'%(params['noofsigma'],params['noofsigmadhz'],params['dynamic_dhz'],params['idea'],params['unsafebuffer'],params['reg_lipschitz']))
        f.write('What I want to write: %s'%(params['quote']))
        f.close()
        utils.init_logging(logdir)#record started!
        log.info('Training safe set MPC with params...')#at the very very start
        log.info(pprint.pformat(params))#just to pretty print all the parameters!
        logger = utils.EpochLogger(logdir)#a kind of dynamic logger?

        env = utils.make_env(params)#spb, reacher, etc.#around line 148 in utils
        #The result is to have env=SimplePointBot in spb
        # Setting up encoder, around line 172 in utils, get all the parts equipped!
        light=params['light']
        #if light=='normal':
        #modules = utils.make_modules(params, ss=True, val=True, dyn=True, gi=True, constr=True)
        modules = utils.make_modulessafety(params, ss=True, val=True, dyn=True, gi=True, constr=True, cbfd=True)
        #modules = utils.make_modulessafetyexpensive(params, ss=True, val=True, dyn=True, gi=True, constr=True, cbfd=True)#forever banned!
        #modules = utils.make_modulessafetyexpensive2(params, ss=True, val=True, dyn=True, gi=True, constr=True, cbfd=True,dyn2=True)#forever banned!
        #the result is to set up the encoder, etc.
        safe_set = modules['ss']
        constraint_function = modules['constr']
        #elif light=='light':
        #modules = utils.make_modulessafetylight(params, val=True, dyn=True, gi=True, cbfd=True)#turn out to be useless
        encoder = modules['enc']#it is a value in a dictionary, uh?
        dynamics_model = modules['dyn']
        value_func = modules['val']
        goal_indicator = modules['gi']
        cbfdot_function = modules['cbfd']
        #encoder2 = modules['enc2']  # it is a value in a dictionary, uh?
        #dynamics_model2 = modules['dyn2']
        # Populate replay buffer
        #the following is loading replay buffer, rather than loading trajectories
        #replay_buffer = utils.load_replay_buffer(params, encoder)#around line 123 in utils.py
        replay_buffer_success = utils.load_replay_buffer_success(params, encoder)#around line 123 in utils.py
        replay_buffer_unsafe = utils.load_replay_buffer_unsafe(params, encoder)#around line 123 in utils.py        
        trainer = MPCTrainer(env, params, modules)#so that we can train MPC!

        #trainer.initial_train(replay_buffer)#initialize all the parts!
        trainer.initial_train_m2(replay_buffer_success,replay_buffer_unsafe)#initialize all the parts!

        log.info("Creating policy")
        #policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,
                                #constraint_function, goal_indicator, params)
        policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,
                                constraint_function, goal_indicator, cbfdot_function, params)
        #policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,#forever banned!
                                #constraint_function, goal_indicator, cbfdot_function, encoder2,params)
        #policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,#forever banned!
                                #constraint_function, goal_indicator, cbfdot_function, encoder2,dynamics_model2, params)
        num_updates = params['num_updates']#default 25
        traj_per_update = params['traj_per_update']#default 10

        losses = {}
        avg_rewards = []
        std_rewards = []
        all_rewards = []
        constr_viols = []
        all_action_rands = []
        constr_viols_cbf = []
        constr_viols_cbf2 = []
        task_succ = []
        n_episodes = 0

        tp, fp, fn, tn, tpc, fpc, fnc, tnc = 0, 0, 0, 0, 0, 0, 0, 0

        reward_type=params['reward_type']
        #print('reward_type',reward_type)
        conservative=params['conservative']
        #print('conservative',conservative)
        action_type=params['action_type']
        cbfalpha=0.2#exponential averaging for CBF
        dhd=0.135#0.13855#how much is the dhd
        dhz=params['dhz']#0#0.00285#0.000545#
        gradh2z=lambda nextobs: cbfdot_function(nextobs, True)
        gradjh2z=lambda obs: torch.norm(jacobian(gradh2z,obs,create_graph=True))
        sth=params['stepstohell']
        for i in range(num_updates):#default 25 in spb
            #log.info('current dhz: %f'%(params['dhz']))I think there is no need to update
            update_dir = os.path.join(logdir, "update_%d" % i)#create the corresponding folder!
            os.makedirs(update_dir)#mkdir!
            update_rewards = []

            # Collect Data
            
            for j in range(traj_per_update):#default 10 in spb
                log.info("Collecting trajectory %d for update %d" % (j, i))
                transitions = []

                obs = np.array(env.reset())#the obs seems to be the observation as image rather than obstacle
                #obs,obs_relative = np.array(env.reset()) #can I do this? # the obs seems to be the observation as image rather than obstacle
                policy.reset()#self.mean, self.std = None, None
                done = False

                # Maintain ground truth info for plotting purposes
                #movie_traj = [{'obs': obs.reshape((-1, 3, 64, 64))[0]}]#a dict
                movie_traj = [{'obs': obs.reshape((-1, 3, 64, 64))[0]}]  # a dict
                #movie_traj_relative = [{'obs_relative': obs_relative.reshape((-1, 3, 64, 64))[0]}]  # a dict
                traj_rews = []#rews: rewards
                traj_action_rands=[]
                constr_viol = False
                succ = False
                action_rand=False
                constr_viol_cbf = False
                constr_viol_cbf2 = False
                
                for k in trange(params['horizon']):#default 100 in spb#This is MPC
                    #print('obs.shape',obs.shape)(3,64,64)
                    #print('env.state',env.state)#env.state [35.44344669 54.30340498]
                    #if action_type=='random':
                        #action = policy.act(obs / 255)#the CEM (candidates, elites, etc.) is in here
                    #elif action_type=='zero':
                        #action = policy.actzero(obs/255)
                    #storch=ptu.torchify(env.state)#state torch
                    #action,tp,fp,fn,tn,tpc,fpc,fnc,tnc = policy.actcbfd(obs/255,env.state,tp,fp,fn,tn,tpc,fpc,fnc,tnc)
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdcircle(obs / 255, env.state, tp, fp, fn, tn, tpc,
                                                                                #fpc, fnc, tnc)
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarecircle(obs / 255, env.state, tp, fp, fn, tn,tpc,fpc, fnc, tnc)
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarelatent(obs / 255, env.state, tp, fp, fn, tn,tpc,fpc, fnc, tnc)
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarelatentplana(obs / 255, env.state, tp, fp,#obs_relative / 255, env.state, tp, fp,#
                                                                                            #fn, tn, tpc, fpc, fnc, tnc)
                    #action,randflag= policy.actcbfdsquarelatentplanareacher(obs / 255)#
                    action,randflag= policy.actcbfdsquarelatentplanareacher(obs / 255,params['dhz'])#
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarelatentplananogoal(obs_relative / 255, env.state, tp, fp,#obs / 255, env.state, tp, fp,
                                                                                            #fn, tn, tpc, fpc, fnc, tnc)
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarelatentplananogoaldense(obs / 255, env.state, tp, fp, fn, tn, tpc, fpc, fnc, tnc)#not finished yet!
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarelatentplana(obs_relative / 255, env.state, tp,
                                                                                                #fp,fn, tn, tpc, fpc, fnc, tnc)
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarelatentplananogoaldense(obs_relative / 255, env.state, tp, fp,
                                                                                            #fn, tn, tpc, fpc, fnc, tnc)#not finished yet!                                                                             
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarelatentplanaexpensive(obs / 255, env.state, tp,
                                                                                                #fp,#forever banned! forever obsolete
                                                                                                #fn, tn, tpc, fpc, fnc, tnc)
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarelatentplanaexpensive2(obs / 255, env.state, tp,
                                                                                                #fp,fn, tn, tpc, fpc, fnc, tnc,obs_relative/255)
                    #action, tp, fp, fn, tn, tpc, fpc, fnc, tnc = policy.actcbfdsquarelatentplanb(obs_relative / 255, env.state, tp,
                                                                                                #fp,
                                                                                                #fn, tn,
                                                                                                #tpc, fpc, fnc, tnc)
                    # the CEM (candidates, elites, etc.) is in here
                    #next_obs, reward, done, info = env.step(action)#saRSa#the info is the extra in the reacher wrapper!
                    #next_obs, reward, done, info = env.step(action)#for reacher, it is step according to the naming issue. But it is actually the stepsafety # env.stepsafety(action)  # 63 in simple_point_bot.py
                    #next_obs, reward, done, info = env.stepsafety(action)#applies to pushing and spb  # 63 in simple_point_bot.py
                    next_obs, reward, done, info = env.stepsafety2(action)#applies to pushing and spb  # 63 in simple_point_bot.py
                    #next_obs = np.array(next_obs)#to make this image a numpy array
                    #next_obs, reward, done, info,next_obs_relative = env.stepsafety_relative(action)  # 63 in simple_point_bot.py
                    next_obs = np.array(next_obs) #relative or not? # to make this image a numpy array
                    #next_obs_relative = np.array(next_obs_relative)  # relative or not? # to make this image a numpy array
                    movie_traj.append({'obs': next_obs.reshape((-1, 3, 64, 64))[0]})  # add this image
                    #movie_traj_relative.append({'obs_relative': next_obs_relative.reshape((-1, 3, 64, 64))[0]}) #relative or not # add this image
                    traj_rews.append(reward)#reward is either 0 or 1!
                    constr = info['constraint']#its use is seen a few lines later
                    rfn=not randflag#rfn means rand flag not#rfn=0 means action not random, rfn=1 means the action is random
                    action_rand=randflag
                    #constr_cbf = rfn*info['constraint']#(1-randflag)*info['constraint']#its use is seen a few lines later
                    if len(traj_action_rands)==0:
                        constr_cbf = rfn*info['constraint']#3when rfn=0, this means that the action is already randomly chosen, CBF has found no solution!
                        constr_cbf2=constr_cbf#constr_cbf=0 means that CBF has correctly detected the danger, but the action (indeed including recovery actions) sucks!
                    else:
                        constr_cbf = rfn*info['constraint']#(1-traj_action_rands[-1]) is the rfn of last step!
                        constr_cbf2 = rfn*info['constraint'] or (1-traj_action_rands[-1])*info['constraint']#one previous step buffer
                        #constr_cbf2 = rfn*info['constraint'] and (1-traj_action_rands[-1])#one previous step buffer
                    traj_action_rands.append(action_rand)#this is really a small bug/inadequacy!
                    hvo=info['hvo']#
                    hvn=info['hvn']#
                    hvd=info['hvd']#,#hvd for h value difference
                    ns=info['next_state']
                    '''
                    transition = {'obs': obs, 'action': action, 'reward': reward,#sARSa
                                'next_obs': next_obs, 'done': done,
                                'constraint': constr, 'safe_set': 0, 'on_policy': 1}
                    '''
                    if params['push_cbf_strategy']==1:
                        transition = {'obs': obs, 'action': action, 'reward': reward,
                                'next_obs': next_obs, 'done': done,  # this is a dictionary
                                'constraint': constr, 'safe_set': 0,
                                'on_policy': 1,
                                'rdo': info['rdo'].tolist(),#rdo for relative distance old
                                'rdn': info['rdn'].tolist(),#rdn for relative distance new
                                'hvo': hvo,#hvo for h value old
                                'hvn': hvn,#hvn for h value new
                                'hvd': hvd,
                                'state': info['state'].tolist(),
                                'next_state': info['next_state'].tolist()
                                }  # add key and value into it!
                    elif params['push_cbf_strategy']==2:
                        hvoef=info['hvoef']#
                        hvnef=info['hvnef']#
                        hvdef=info['hvdef']#,#hvd for h value difference
                        transition = {'obs': obs, 'action': action, 'reward': reward,
                                'next_obs': next_obs, 'done': done,  # this is a dictionary
                                'constraint': constr, 'safe_set': 0,
                                'on_policy': 1,
                                'rdo': info['rdo'].tolist(),#rdo for relative distance old
                                'rdn': info['rdn'].tolist(),#rdn for relative distance new
                                'hvo': hvo,#hvo for h value old
                                'hvn': hvn,#hvn for h value new
                                'hvd': hvd,
                                'rdoef': info['rdoef'].tolist(),#rdo for relative distance old
                                'rdnef': info['rdnef'].tolist(),#rdn for relative distance new
                                'hvoef': hvoef,#hvo for h value old
                                'hvnef': hvnef,#hvn for h value new
                                'hvdef': hvdef,
                                'state': info['state'].tolist(),
                                'next_state': info['next_state'].tolist()
                                }  # add key and value into it!

                    transitions.append(transition)

                    if j==0:#just sample one trajectory, other trajectories are basicly replications
                        currentstate=info['state']#the 27 dimensional thing!
                        #print('currentstate',currentstate)#8 dim vector!#the first 2 values are still in configuration space!
                        currentpos=info['rdo']#targetpos-currentstate[2:4]#currentstate[0:2]#now the current pos should have different meaning!
                        ctoobstacle=currentpos-0.4#currentstate[4:6]#
                        ctodistance=ctoobstacle#np.linalg.norm(ctoobstacle)#it will just be the absolute value!#
                        #print('ctoobstacle',ctoobstacle)#the x and y signed distance to obstacle!
                        #print('currentpos',currentpos)#now it is the state space position of the end effector!
                        nextstate=info['next_state']
                        nextpos=info['rdn']##targetpos-nextstate[2:4]#nextstate[0:2]#
                        ntoobstacle=nextpos-0.4#nextstate[4:6]#
                        ntodistance=ntoobstacle#np.linalg.norm(ntoobstacle)#it will just be the absolute value!#
                        #print('nextstate',nextstate)
                        #print('nextpos',nextpos)
                        posdiff=nextpos-currentpos
                        posdiffnorm=np.linalg.norm(posdiff)
                        pdnarray[piece]=posdiffnorm
                        imobs = ptu.torchify(obs).reshape(1, *obs.shape)#it seems that this reshaping is necessary#np.array(frame['obs'])#(transition[key])#seems to be the image?
                        imnextobs = ptu.torchify(next_obs).reshape(1, *obs.shape)#np.array(frame['next_obs'])#(transition[key])#seems to be the image?
                        #imnextobs = ptu.torchify(imnextobs)
                        if params['mean']=='sample':
                            zobs = encoder.encode(imobs/255)#in latent space now!#even
                            znextobs = encoder.encode(imnextobs/255)#in latent space now!#even
                        elif params['mean']=='mean' or params['mean']=='meancbf':
                            #log.info('it is using the mean output from the encoder!')#checked!!
                            zobs = encoder.encodemean(imobs/255)#in latent space now!#really zero now! That's what I  want!
                            znextobs = encoder.encodemean(imnextobs/255)#in latent space now!#really zero now! That's what I  want!
                        imdiff1=imnextobs/255-imobs/255
                        #print('imdiff1',imdiff1)#3 channel image!
                        imagediff=ptu.to_numpy(imdiff1)#next_obs-obs#frame['next_obs']-frame['obs']
                        #imagediffnorm=np.linalg.norm(imagediff)#
                        imagediffnormal=np.linalg.norm(imagediff)#imagediffnorm/255#
                        zdiff=ptu.to_numpy(znextobs-zobs)
                        zdiffnorm=np.linalg.norm(zdiff)
                        hobs=cbfdot_function(zobs,already_embedded=True)##cbfd(zobs_mean)
                        hnextobs=cbfdot_function(znextobs,already_embedded=True)#cbfd(znext_obs_mean)
                        #log.info('hobs: %f, hnextobs: %f'%(hobs,hnextobs))
                        dtzobs=gradjh2z(zobs)*dhd
                        bzuop=hobs-dtzobs
                        dtznextobs=gradjh2z(znextobs)*dhd
                        bzunop=hnextobs-dtznextobs
                        log.info('hobs: %f, hnextobs: %f, dtzobs: %f, dtznextobs: %f'%(hobs,hnextobs,dtzobs,dtznextobs))
                        qzuop=bzuop-dhz
                        qzunop=bzunop-dhz
                        qdiff=ptu.to_numpy(qzunop-qzuop)
                        qdiffnorm=np.linalg.norm(qdiff)
                        hdiff=ptu.to_numpy(hnextobs-hobs)
                        hdiffnorm=np.linalg.norm(hdiff)
                        if posdiffnorm<1e-3:# or imagediffnormal<1e-3:#5e-4:#2e-3:#1e-2:#posdiffnorm<=1e-4:#otherwise it is meaningless!
                            imagediffnormal=0
                            zdiffnorm=0
                            hdiffnorm=0
                            qdiffnorm=0
                        slopexyp=imagediffnormal/(posdiffnorm+eps)
                        slopeyzp=zdiffnorm/(imagediffnormal+eps)
                        slopezhp=hdiffnorm/(zdiffnorm+eps)
                        slopeyhp=hdiffnorm/(imagediffnormal+eps)
                        slopexhp=hdiffnorm/(posdiffnorm+eps)
                        slopezqp=qdiffnorm/(zdiffnorm+eps)
                        slopeyqp=qdiffnorm/(imagediffnormal+eps)
                        slopexqp=qdiffnorm/(posdiffnorm+eps)
                        slopexy[piece]=slopexyp
                        slopeyz[piece]=slopeyzp
                        slopezh[piece]=slopezhp
                        slopeyh[piece]=slopeyhp
                        slopexh[piece]=slopexhp
                        slopezq[piece]=slopezqp
                        slopeyq[piece]=slopeyqp
                        slopexq[piece]=slopexqp
                        qzuno[piece]=qzunop
                        lipxy=max(lipxy,slopexyp)
                        lipyz=max(lipyz,slopeyzp)
                        lipzh=max(lipzh,slopezhp)
                        lipyh=max(lipyh,slopeyhp)
                        lipxh=max(lipxh,slopexhp)
                        lipzq=max(lipzq,slopezhp)
                        lipyq=max(lipyq,slopeyhp)
                        lipxq=max(lipxq,slopexhp)
                        gammadyn=min(gammadyn,qzunop)
                        pdn=max(pdn,posdiffnorm)
                        if np.abs(ntodistance)<=0.30 and np.abs(ntodistance)>=0.25:#the new safe region I pick!#ntodistance>=0.20:#ntodistance<=0.09 and ntodistance>=0.07:#
                            slopexys[piece]=slopexyp
                            slopeyzs[piece]=slopeyzp
                            slopezhs[piece]=slopezhp
                            slopeyhs[piece]=slopeyhp
                            slopexhs[piece]=slopexhp
                            slopezqs[piece]=slopezqp
                            slopeyqs[piece]=slopeyqp
                            slopexqs[piece]=slopexqp
                            qzunos[piece]=qzunop
                            lipxysafe=max(lipxysafe,slopexyp)
                            lipyzsafe=max(lipyzsafe,slopeyzp)
                            lipzhsafe=max(lipzhsafe,slopezhp)
                            lipyhsafe=max(lipyhsafe,slopeyhp)
                            lipxhsafe=max(lipxhsafe,slopexhp)
                            lipzqsafe=max(lipzqsafe,slopezqp)
                            lipyqsafe=max(lipyqsafe,slopeyqp)
                            lipxqsafe=max(lipxqsafe,slopexqp)
                            gammadyns=min(gammadyns,qzunop)
                            pdnsafe=max(pdnsafe,posdiffnorm)
                            log.info('piece:%d,sxysp:%f,syzsp:%f,szhsp:%f,syhsp:%f,sxhsp:%f,szqsp:%f,syqsp:%f,sxqsp:%f,pdnorm:%f,qzunos:%f,ntodistance:%f' % (piece,slopexyp,slopeyzp,slopezhp,slopeyhp,slopexhp,slopezqp,slopeyqp,slopexqp,posdiffnorm,qzunop,ntodistance))
                            log.info('piece:%d,lxys:%f,lyzs:%f,lzhs:%f,lyhs:%f,lxhs:%f,lzqs:%f,lyqs:%f,lxqs:%f,pdns:%f,gammadyns:%f' % (piece,lipxysafe,lipyzsafe,lipzhsafe,lipyhsafe,lipxhsafe,lipzqsafe,lipyqsafe,lipxqsafe,pdnsafe,gammadyns))
                        elif np.abs(ntodistance)<=0.15 and np.abs(ntodistance)>=0.10:##np.abs(ntodistance)<=0.10:#unsafe#ntodistance<=0.06:#it is good to use distance to judge safety!
                            slopexyu[piece]=slopexyp
                            slopeyzu[piece]=slopeyzp
                            slopezhu[piece]=slopezhp
                            slopeyhu[piece]=slopeyhp
                            slopexhu[piece]=slopexhp
                            lipxyunsafe=max(lipxyunsafe,slopexyp)
                            lipyzunsafe=max(lipyzunsafe,slopeyzp)
                            lipzhunsafe=max(lipzhunsafe,slopezhp)
                            lipyhunsafe=max(lipyhunsafe,slopeyhp)
                            lipxhunsafe=max(lipxhunsafe,slopexhp)
                            log.info('piece:%d,sxyusp:%f,syzusp:%f,szhusp:%f,syhusp:%f,sxhusp:%f,pdnorm:%f,ntodistance:%f' % (piece,slopexyp,slopeyzp,slopezhp,slopeyhp,slopexhp,posdiffnorm,ntodistance))
                            log.info('piece:%d,lxyus:%f,lyzus:%f,lzhus:%f,lyhus:%f,lxhus:%f' % (piece,lipxyunsafe,lipyzunsafe,lipzhunsafe,lipyhunsafe,lipxhunsafe))
                        else:
                            log.info('piece:%d,sxyp:%f,syzp:%f,szhp:%f,syhp:%f,sxhp:%f,szqp:%f,syqp:%f,sxqp:%f,pdnorm:%f,qzuno:%f,ntodistance:%f' % (piece,slopexyp,slopeyzp,slopezhp,slopeyhp,slopexhp,slopezqp,slopeyqp,slopexqp,posdiffnorm,qzunop,ntodistance))
                            log.info('piece:%d,lipxy:%f,lipyz:%f,lipzh:%f,lipyh:%f,lipxh:%f,lipzq:%f,lipyq:%f,lipxq:%f,pdn:%f,gammadyn:%f' % (piece,lipxy,lipyz,lipzh,lipyh,lipxh,lipzq,lipyq,lipxq,pdn,gammadyn))
                        piece+=1

                    obs = next_obs#don't forget this step!
                    #print('obs.shape',obs.shape)#(3, 3, 64, 64)
                    #obs_relative = next_obs_relative  # don't forget this step!
                    constr_viol = constr_viol or info['constraint']#a way to update constr_viol#either 0 or 1
                    constr_viol_cbf = constr_viol_cbf or constr_cbf#a way to update constr_viol#either 0 or 1
                    constr_viol_cbf2 = constr_viol_cbf2 or constr_cbf2#a way to update constr_viol#either 0 or 1
                    succ = succ or reward == 0#as said in the paper, reward=0 means success!

                    
                    #Now, I should do the evaluation!
                    obseval= ptu.torchify(obs).reshape(1, *obs.shape)#it seems that this reshaping is necessary
                    #obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing#pay attention to its shape!#prepare to be used!
                    #embeval = encoder.encode(obseval)#in latent space now!
                    if params['mean']=='sample':
                        embeval = encoder.encode(obseval/255)#encoder.encode(obseval)#in latent space now!#even
                    elif params['mean']=='mean' or params['mean']=='meancbf':
                        embeval = encoder.encodemean(obseval/255)#encoder.encodemean(obseval)#in latent space now!#really zero now! That's what I  want!
                        #embeval2 = encoder.encodemean(obseval)#in latent space now!
                    #print('emb.shape',emb.shape)#torch.Size([1, 32])
                    #cbfdot_function.predict()
                    cbfpredict = cbfdot_function(embeval,already_embedded=True)#
                    '''
                    cbfgt=hvn
                    if (cbfpredict>=0) and (cbfgt>=0):
                        tn+=1
                    elif (cbfpredict>=0) and (cbfgt<0):
                        fn+=1
                    elif (cbfpredict<0) and (cbfgt>=0):
                        fp+=1
                    elif (cbfpredict<0) and (cbfgt<0):
                        tp+=1
                    tncvalue=0.3**2-0.4**2+1e-3#FOR PUSHING!#0.05**2-0.055**2+1e-4#for reacher!#0.05**2-0.06**2+1e-4#
                    if (cbfpredict>=0) and (cbfgt>=tncvalue):
                        tnc+=1
                    elif (cbfpredict>=0) and (cbfgt<tncvalue):
                        fnc+=1
                    elif (cbfpredict<0) and (cbfgt>=tncvalue):
                        fpc+=1
                    elif (cbfpredict<0) and (cbfgt<tncvalue):
                        tpc+=1
                    #log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,s_x:%f,s_y:%f,c_viol:%d,c_viol_cbf:%d,c_viol_cbf2:%d,a_rand:%d' % (tp, fp, fn, tn, tpc, fpc, fnc, tnc,ns[0],ns[1],constr_viol,constr_viol_cbf,constr_viol_cbf2,action_rand))
                    #the evaluation phase ended
                    '''
                    log.info('s_x:%f,s_y:%f,c_viol:%d,c_viol_cbf:%d,c_viol_cbf2:%d,a_rand:%d,cbfpredict:%f' % (ns[0],ns[1],constr_viol,constr_viol_cbf,constr_viol_cbf2,action_rand,cbfpredict))
                    if done:
                        break
                transitions[-1]['done'] = 1#change the last transition to success/done!
                traj_reward = sum(traj_rews)#total reward, should be >=-100/-150
                #EpRet is episode reward, EpLen=Episode Length, EpConstr=Episode constraints
                logger.store(EpRet=traj_reward, EpLen=k+1, EpConstr=float(constr_viol), EpConstrcbf=float(constr_viol_cbf), EpConstrcbf2=float(constr_viol_cbf2))
                all_rewards.append(traj_rews)#does it use any EpLen?
                all_action_rands.append(traj_action_rands)
                constr_viols.append(constr_viol)#whether this 100-length traj violate any constraints, then compute the average
                constr_viols_cbf.append(constr_viol_cbf)#
                constr_viols_cbf2.append(constr_viol_cbf2)#
                task_succ.append(succ)
                #save the result in the gift form!
                pu.make_movie(movie_traj, file=os.path.join(update_dir, 'trajectory%d.gif' % j))
                #pu.make_movie_relative(movie_traj_relative, file=os.path.join(update_dir, 'trajectory%d_relative.gif' % j))

                log.info('    Cost: %d, constraint violation: %d, cv_cbf: %d, cv_cbf2: %d' % (traj_reward,constr_viol,constr_viol_cbf,constr_viol_cbf2))#see it in the terminal!
                #log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d' % (tp, fp, fn, tn, tpc, fpc, fnc, tnc))
                in_ss = 0
                rtg = 0
                '''
                for transition in reversed(transitions):#old way!
                    if transition['reward'] > -1:
                        in_ss = 1
                    transition['safe_set'] = in_ss
                    transition['rtg'] = rtg

                    rtg = rtg + transition['reward']
                '''
                if params['ways']==1:#old way#data generated by new way can still be used by the old way
                    for transition in reversed(transitions):
                        if transition['reward'] > -1:
                            in_ss = 1
                        transition['safe_set'] = in_ss
                        transition['rtg'] = rtg

                        rtg = rtg + transition['reward']
                elif params['ways']==2:#new way
                    for n in reversed(range(params['horizon'])):#n is from 99 to 0 inclusive
                        frame=transitions[n]
                        if frame['reward'] >= 0:
                            ss = 1
                        #along the way of the trajectroy, the trajectory is safe
                        frame['safe_set'] = ss#is this dynamic programming?
                        frame['rtg'] = rtg#the reward to goal at each frame!#I think this is good
                        #add a key value pair to the trajectory(key='rtg', value=rtg
                        rtg = rtg + frame['reward']
                        #now the new things start!
                        if n>=2:#1:#frame[0]'s constraint is always 0! initial condition is always safe!
                            frameprevious=transitions[n-1]
                            if (frame['constraint']-frameprevious['constraint'])>0 and frame['constraint']>1e-6:#to avoid numerical issues!
                                frameprevious['constraint']=frame['constraint']-1/sth#it is still self supervised!#

                #replay_buffer.store_transitions(transitions)#replay buffer online training
                if not constr_viol:
                    replay_buffer_success.store_transitions(transitions)
                else:
                    replay_buffer_unsafe.store_transitions(transitions)
                update_rewards.append(traj_reward)

            mean_rew = float(np.mean(update_rewards))
            std_rew = float(np.std(update_rewards))
            avg_rewards.append(mean_rew)
            std_rewards.append(std_rew)
            log.info('Iteration %d average reward: %.4f' % (i, mean_rew))
            pu.simple_plot(avg_rewards, std=std_rewards, title='Average Rewards',
                        file=os.path.join(logdir, 'rewards%dthepoch%1.5f.pdf'%(i,np.mean(constr_viols))),
                        ylabel='Average Reward', xlabel='# Training updates')

            logger.log_tabular('Epoch', i)
            logger.log_tabular('TrainEpisodes', n_episodes)
            logger.log_tabular('TestEpisodes', traj_per_update)
            logger.log_tabular('EpRet')
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('EpConstr', average_only=True)
            logger.log_tabular('EpConstrcbf', average_only=True)
            logger.log_tabular('EpConstrcbf2', average_only=True)
            logger.log_tabular('ConstrRate', np.mean(constr_viols))
            logger.log_tabular('ConstrcbfRate', np.mean(constr_viols_cbf))
            logger.log_tabular('Constrcbf2Rate', np.mean(constr_viols_cbf2))
            logger.log_tabular('SuccRate', np.mean(task_succ))
            logger.dump_tabular()
            n_episodes += traj_per_update#10 by default

            # Update models

            #trainer.update(replay_buffer, i)#online training, right?
            #episodiccbfdhz=trainer.update(replay_buffer, i,replay_buffer_unsafe)#online training, right?
            episodiccbfdhz=trainer.update_m2(replay_buffer_success, i,replay_buffer_unsafe)#online training, right?#it now only bears the meaning of dhz!
            if params['dynamic_dhz']=='yes':
                dhzoriginal=params['dhz']
                #log.info('old dhz: %f'%(dhzoriginal))#not needed, as it is already printed at the begining of each episode
                params['dhz']=(1-cbfalpha)*dhzoriginal+cbfalpha*episodiccbfdhz#*params['noofsigmadhz']*(2-params['cbfdot_thresh'])
            log.info('new dhz: %f'%(params['dhz']))#if dynamic_dhz=='no', then it will be still the old dhz
            np.save(os.path.join(logdir, 'rewards.npy'), all_rewards)
            np.save(os.path.join(logdir, 'constr.npy'), constr_viols)
            np.save(os.path.join(logdir, 'constrcbf.npy'), constr_viols_cbf)
            np.save(os.path.join(logdir, 'constrcbf2.npy'), constr_viols_cbf2)
            np.save(os.path.join(logdir, 'action_rands.npy'), all_action_rands)
            np.save(os.path.join(logdir, 'tasksuccess.npy'), task_succ)
        #save after one seed!
        np.save(os.path.join(logdir, 'slopexy.npy'), slopexy)
        np.save(os.path.join(logdir, 'slopeyz.npy'), slopeyz)
        np.save(os.path.join(logdir, 'slopezh.npy'), slopezh)
        np.save(os.path.join(logdir, 'slopeyh.npy'), slopeyh)
        np.save(os.path.join(logdir, 'slopexh.npy'), slopexh)
        np.save(os.path.join(logdir, 'slopezq.npy'), slopezq)
        np.save(os.path.join(logdir, 'slopeyq.npy'), slopeyq)
        np.save(os.path.join(logdir, 'slopexq.npy'), slopexq)
        np.save(os.path.join(logdir, 'qzuno.npy'), qzuno)
        np.save(os.path.join(logdir, 'pdn.npy'), pdnarray)
        np.save(os.path.join(logdir, 'slopexys.npy'), slopexys)
        np.save(os.path.join(logdir, 'slopeyzs.npy'), slopeyzs)
        np.save(os.path.join(logdir, 'slopezhs.npy'), slopezhs)
        np.save(os.path.join(logdir, 'slopeyhs.npy'), slopeyhs)
        np.save(os.path.join(logdir, 'slopexhs.npy'), slopexhs)
        np.save(os.path.join(logdir, 'slopezqs.npy'), slopezqs)
        np.save(os.path.join(logdir, 'slopeyqs.npy'), slopeyqs)
        np.save(os.path.join(logdir, 'slopexqs.npy'), slopexqs)
        np.save(os.path.join(logdir, 'qzunos.npy'), qzunos)
        np.save(os.path.join(logdir, 'slopexyu.npy'), slopexyu)
        np.save(os.path.join(logdir, 'slopeyzu.npy'), slopeyzu)
        np.save(os.path.join(logdir, 'slopezhu.npy'), slopezhu)
        np.save(os.path.join(logdir, 'slopeyhu.npy'), slopeyhu)
        np.save(os.path.join(logdir, 'slopexhu.npy'), slopexhu)
        params['seed']=params['seed']+1#m+1#
        #utils.init_logging(logdir)#record started!
        #logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M:%S',filename=os.path.join(logdir, 'logjianning.txt'),filemode='w')