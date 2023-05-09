
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/cuijin/Project6remote/latent-space-safe-sets')

from latentsafesets.policy import CEMSafeSetPolicy#this is the class!
import latentsafesets.utils as utils
import latentsafesets.utils.plot_utils as pu
#from latentsafesets.utils.arg_parser import parse_args
from latentsafesets.utils.arg_parser_reacher import parse_args
from latentsafesets.rl_trainers import MPCTrainer
import latentsafesets.utils.pytorch_utils as ptu

import os
import logging
from tqdm import trange#mainly for showing the progress bar
import numpy as np
import pprint

from latentsafesets.modules import VanillaVAE, CBFdotEstimator,CBFdotEstimatorlatentplana
from datetime import datetime
#provides a capability to “pretty-print” arbitrary Python data structures in a form that can be used as input to the interpreter
#log = logging.getLogger("main")#some logging stuff

from latentsafesets.utils.teacher import ReacherConstraintdense1Teacher, ReacherConstraintdense2Teacher
import sympy

if __name__ == '__main__':
    params = parse_args()#get the parameters from parse_args, see arg_parser.py
    # Misc preliminaries
    repeattimes=1#params['repeat_times']#
    initdhz=params['dhz']
    traj_per_update = 200#params['traj_per_update']#default 10
    params['horizon']=320#500#400#
    slopexy=np.zeros((traj_per_update*params['horizon']))
    slopeyz=np.zeros((traj_per_update*params['horizon']))
    slopezh=np.zeros((traj_per_update*params['horizon']))
    slopeyh=np.zeros((traj_per_update*params['horizon']))
    piece=0#which piece of trajectory? this piece
    eps=1e-10
    for m in range(repeattimes):
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
        f.write('r_{thres}=1.2\t')
        f.write('action_type: %s\t'%(params['action_type']))
        #f.write('conservativeness: %s, reward_type: %s\t'%(params['conservative'],params['reward_type']))
        f.write('conservativeness: %s, reward_type: %s, lightness: %s\n'%(params['conservative'],params['reward_type'],params['light']))
        f.write('reduce_horizon: %s, 0 or 1: %s\n'%(params['reduce_horizon'],params['zero_one']))
        f.write('episodic train cbf or not: %s\n'%(params['train_cbf']))
        f.write('mean: %s, dhz: %f, dhdmax: %f\n'%(params['mean'],params['dhz'],params['dhdmax']))
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
        #replay_buffer = utils.load_replay_buffer_relative(params, encoder)  # around line 123 in utils.py
        #replay_buffer2 = utils.load_replay_buffer_relative(params, encoder2)  # around line 123 in utils.py
        #replay_buffer = utils.load_replay_buffer_relative_expensive2(params, encoder, encoder2)  # around line 123 in utils.py
        replay_buffer = utils.load_replay_buffer_lipschitz(params, encoder)#around line 123 in utils.py
        '''
        if params['unsafebuffer']=='yes':
            replay_buffer_unsafe = utils.load_replay_buffer_unsafe(params, encoder)#around line 123 in utils.py
            log.info('unsafe buffer!')
        else:
            replay_buffer_unsafe=replay_buffer
            log.info('the same buffer!')#have checked np.random.randint, it is completely random! This is what I want!
        '''
        replay_buffer_unsafe=None
        trainer = MPCTrainer(env, params, modules)#so that we can train MPC!

        #trainer.initial_train(replay_buffer,replay_buffer_unsafe)#initialize all the parts!

        log.info("Creating policy")
        #policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,
                                #constraint_function, goal_indicator, params)
        policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,
                                constraint_function, goal_indicator, cbfdot_function, params)
        #policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,#forever banned!
                                #constraint_function, goal_indicator, cbfdot_function, encoder2,params)
        #policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,#forever banned!
                                #constraint_function, goal_indicator, cbfdot_function, encoder2,dynamics_model2, params)
        num_updates = 1#params['num_updates']#default 25
        

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
        for i in range(num_updates):#default 25 in spb
            #if i==0:
                #teacher=ReacherConstraintdense1Teacher(env,noisy=False)
            #elif i==1:
                #teacher=ReacherConstraintdense2Teacher(env,noisy=False)
            teacher=ReacherConstraintdense2Teacher(env,noisy=False)
            log.info('current dhz: %f'%(params['dhz']))
            update_dir = os.path.join(logdir, "update_%d" % i)#create the corresponding folder!
            datasave_dir = os.path.join(update_dir, "ReacherConstraintdense2")#create the corresponding folder!
            os.makedirs(update_dir)#mkdir!
            os.makedirs(datasave_dir)#mkdir!
            update_rewards = []


            #teacher = teacher(env)#, noisy=noisy)#SimplePointBotTeacher, or ConstraintTeacher,
            '''
            demonstrations = []#an empty list
            #ro=0.04#0.05-1
            for i in range(n):#50/250
                print('data_dir',data_dir)
                if data_dir=='ReacherConstraintdense1' or data_dir=='ReacherConstraintdense2':
                    #print('teacherenter',teacher)
                    


                    #print('s1t2',s1t2)#print('s1t1',s1[t1])
                    traj = teacher.generate_trajectorysafety_dense(xa=s0t1,ya=s0t2,xa2=s1t1,ya2=s1t2,angled=angled)
                else:
                    traj = teacher.generate_trajectorysafety()#line 33 in teacher.py#100 transitions
                reward = sum([frame['reward'] for frame in traj])#traj is a list of dictionaries
                #why not directly use rtg[0]?
                print('Trajectory %d, Reward %d' % (i, reward))
                demonstrations.append(traj)#traj is one piece of trajectories
                utils.save_trajectory(traj, file, i)#86 in utils.py#save 1 traj having 100 steps

                if i < 50 and logdir is not None:
                    pu.make_movie(traj, os.path.join(logdir, '%s_%d.gif' % (data_dir, i)))#do I add relative here or in other place?
            '''


            # Collect Data
            for j in range(traj_per_update):#default 10 in spb


                angled=-np.pi + 1.5*np.pi*(j+1)/traj_per_update#n#here n is traj_per_update##np.pi/2 - 1.5*np.pi*i/n#d means desired
                if type(teacher)==ReacherConstraintdense1Teacher:#'ReacherConstraintdense1':
                    radius=0.045#0.05#0.04#
                else:
                    radius=0.12
                    if j<2:
                        radius=(0.11+0.005*j)
                xinc=radius*np.cos(angled)
                yinc=radius*np.sin(angled)
                xbase=-0.13*np.sqrt(0.75)
                ybase=0.065
                xtotal=xbase+xinc
                ytotal=ybase+yinc
                t1,t4=sympy.symbols("t1,t4", real=True)
                length=0.12
                eq1=sympy.Eq(length*(sympy.cos(t4)+sympy.cos(t1)),xtotal)#-0.13*np.sqrt(0.75))#-0.169705)#
                eq2=sympy.Eq(length*(sympy.sin(t4)+sympy.sin(t1)),ytotal)#0.065)#0.169705)#
                solt=sympy.solve([eq1, eq2])
                #print('x',x)#print('y',y)#print('solt[0]',solt[0])#print('solt[1]',solt[1])
                s0=solt[0]#
                s1=solt[1]
                #print('s0',s0)#print('s1',s1)print('s0t4',s0[t4])
                s0t1=s0[t1]
                s0t2=s0[t4]-s0[t1]
                if s0t1<-np.pi:
                    s0t1+=2*np.pi
                elif s0t1>np.pi:
                    s0t1-=2*np.pi
                #print('s0t1',s0t1)
                if s0t2<-np.pi:
                    s0t2+=2*np.pi
                elif s0t2>np.pi:
                    s0t2-=2*np.pi
                #print('s0t2',s0t2)#print('s1t1',s1[t1])
                s1t1=s1[t1]
                s1t2=s1[t4]-s1[t1]
                if s1t1<-np.pi:
                    s1t1+=2*np.pi
                elif s1t1>np.pi:
                    s1t1-=2*np.pi
                #print('s1t1',s1t1)
                if s1t2<-np.pi:
                    s1t2+=2*np.pi
                elif s1t2>np.pi:
                    s1t2-=2*np.pi


                log.info("Collecting trajectory %d for update %d" % (j, i))
                transitions = []

                obs = np.array(env.reset())#the obs seems to be the observation as image rather than obstacle
                #obs = np.array(env.reset(s0t1))#just a test
                #obs,obs_relative = np.array(env.reset()) #can I do this? # the obs seems to be the observation as image rather than obstacle
                policy.reset()#self.mean, self.std = None, None
                done = False
                #state = None
                # Maintain ground truth info for plotting purposes
                #movie_traj = [{'obs': obs.reshape((-1, 3, 64, 64))[0]}]#a dict
                movie_traj = [{'obs': obs.reshape((-1, 3, 64, 64))[0]}]  # a dict
                #movie_traj_relative = [{'obs_relative': obs_relative.reshape((-1, 3, 64, 64))[0]}]  # a dict
                traj_rews = []#rews: rewards
                constr_viol = False
                succ = False
                traj_action_rands=[]
                action_rand=False
                constr_viol_cbf = False
                constr_viol_cbf2 = False
                
                for k in trange(params['horizon']):#default 100 in spb#This is MPC
                    #print('obs.shape',obs.shape)(3,64,64)
                    #print('env.state',env.state)#env.state [35.44344669 54.30340498]
                    state=env.current_state
                    #print('self.current_state',env.current_state)#it is an 8-dimensional vector
                    #if env.state is None:
                        #action = teacher.env.action_space.sample().astype(np.float64)#sample between -3 and 3
                    #else:#I think the control is usually either -3 or +3
                        #action = self._expert_control_dense(state, i,xa,ya,xa2,ya2,angled).astype(np.float64)
                    action=teacher._expert_control_dense_lip(state,k,xa=s0t1,ya=s0t2,xa2=s1t1,ya2=s1t2,angled=angled).astype(np.float64)
                    action=np.float32(action)#has to be like this?#this is important!
                    '''
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

                    next_obs, reward, done, info = env.step(action_input)#which one?
                    #action,randflag= policy.actcbfdsquarelatentplanareacher(obs / 255)#,conservative,reward_type)#
                    '''
                    randflag=0
                    # the CEM (candidates, elites, etc.) is in here
                    #next_obs, reward, done, info = env.step(action)#saRSa#the info is the extra in the reacher wrapper!
                    next_obs, reward, done, info = env.step(action)#for reacher, it is step according to the naming issue. But it is actually the stepsafety # env.stepsafety(action)  # 63 in simple_point_bot.py
                    #next_obs, reward, done, info = env.stepsafety(action)#applies to pushing and spb  # 63 in simple_point_bot.py
                    #next_obs = np.array(next_obs)#to make this image a numpy array
                    #next_obs, reward, done, info,next_obs_relative = env.stepsafety_relative(action)  # 63 in simple_point_bot.py
                    next_obs = np.array(next_obs) #relative or not? # to make this image a numpy array
                    #next_obs_relative = np.array(next_obs_relative)  # relative or not? # to make this image a numpy array
                    movie_traj.append({'obs': next_obs.reshape((-1, 3, 64, 64))[0]})  # add this image
                    #movie_traj_relative.append({'obs_relative': next_obs_relative.reshape((-1, 3, 64, 64))[0]}) #relative or not # add this image
                    traj_rews.append(reward)#reward is either 0 or 1!

                    constr = info['constraint']#its use is seen a few lines later

                    rfn=not randflag#rfn means rand flag not
                    action_rand=randflag
                    #constr_cbf = rfn*info['constraint']#(1-randflag)*info['constraint']#its use is seen a few lines later
                    if len(traj_action_rands)==0:
                        constr_cbf = rfn*info['constraint']#
                        constr_cbf2=constr_cbf
                    else:
                        constr_cbf = rfn*info['constraint']#
                        constr_cbf2 = rfn*info['constraint'] or (1-traj_action_rands[-1])*info['constraint']#one previous step buffer
                    traj_action_rands.append(action_rand)
                    hvo=info['hvo']#
                    hvn=info['hvn']#
                    hvd=info['hvd']#,#hvd for h value difference
                    ns=info['next_state']
                    '''
                    transition = {'obs': obs, 'action': action, 'reward': reward,#sARSa
                                'next_obs': next_obs, 'done': done,
                                'constraint': constr, 'safe_set': 0, 'on_policy': 1}
                    '''
                    #print(np.dtype(float(reward)))
                    #print(np.dtype(hvo))
                    #print(np.dtype(hvn))
                    #print(np.dtype(hvd))
                    action64=np.float64(action)
                    transition = {'obs': obs, 'action': tuple(action64), 'reward': float(reward),
                                'next_obs': next_obs, 'done': int(done),  # this is a dictionary
                                'constraint': int(constr), 'safe_set': 0,
                                'on_policy': 1,
                                'rdo': info['rdo'].tolist(),#rdo for relative distance old
                                'rdn': info['rdn'].tolist(),#rdn for relative distance new
                                'hvo': hvo,#hvo for h value old
                                'hvn': hvn,#hvn for h value new
                                'hvd': hvd,
                                'state': info['state'].tolist(),
                                'next_state': info['next_state'].tolist()
                                }  # add key and value into it!
                    '''
                    transition = {'obs': obs, 'action': action, 'reward': reward,
                                'next_obs': next_obs, 'done': done,  # this is a dictionary
                                'constraint': constr, 'safe_set': 0,
                                'on_policy': 1,
                                'rdo': info['rdo'].tolist(),  # rdo for relative distance old
                                'rdn': info['rdn'].tolist(),  # rdn for relative distance new
                                'hvo': info['hvo'],  # hvo for h value old
                                'hvn': info['hvn'],  # hvn for h value new
                                'hvd': info['hvd'],  # hvd for h value difference
                                'state': info['state'].tolist(),
                                'next_state': info['next_state'].tolist(),
                                'state_relative': info['state_relative'].tolist(),
                                'next_state_relative': info['next_state_relative'].tolist(),
                                'obs_relative': obs_relative,
                                'next_obs_relative': next_obs_relative
                                }  # add key and value into it!
                    '''


                    transitions.append(transition)





                    currentstate=info['state']
                    currentpos=currentstate[0:2]
                    nextstate=info['next_state']
                    nextpos=nextstate[0:2]
                    posdiff=nextpos-currentpos
                    posdiffnorm=np.linalg.norm(posdiff)
                    #for key in im_dat:#from obs and next_obs
                        #frame[key] = im_dat[key][j]#the frame is the jth frame in 1 traj
                    #frame['obs'] = im_dat['obs'][j]#the frame is the jth frame in 1 traj
                    #frame['next_obs'] = im_dat['next_obs'][j]#the frame is the jth frame in 1 traj
                    imagediff=next_obs-obs#frame['next_obs']-frame['obs']
                    imagediffnorm=np.linalg.norm(imagediff)
                    imagediffnormal=imagediffnorm/255
                    imobs = ptu.torchify(obs).reshape(1, *obs.shape)#it seems that this reshaping is necessary#np.array(frame['obs'])#(transition[key])#seems to be the image?
                    #zobs_mean, zobs_log_std = self.encoder(imobs[None] / 255)#is it legit?
                    #zobs_mean = zobs_mean.squeeze().detach().cpu().numpy()
                    if params['mean']=='sample':
                        zobs = encoder.encode(imobs/255)#in latent space now!#even
                    elif params['mean']=='mean':
                        zobs = encoder.encodemean(imobs/255)#in latent space now!#really zero now! That's what I  want!
                    imnextobs = ptu.torchify(next_obs).reshape(1, *obs.shape)#np.array(frame['next_obs'])#(transition[key])#seems to be the image?
                    #imnextobs = ptu.torchify(imnextobs)
                    #znext_obs_mean, znext_obs_log_std = self.encoder(imnextobs[None] / 255)#is it legit?
                    #znext_obs_mean = znext_obs_mean.squeeze().detach().cpu().numpy()
                    if params['mean']=='sample':
                        znextobs = encoder.encode(imnextobs/255)#in latent space now!#even
                    elif params['mean']=='mean':
                        znextobs = encoder.encodemean(imnextobs/255)#in latent space now!#really zero now! That's what I  want!
                    zdiff=znextobs-zobs
                    zdiffnorm=np.linalg.norm(zdiff)
                    hobs=cbfdot_function(zobs,already_embedded=True)##cbfd(zobs_mean)
                    hnextobs=cbfdot_function(znextobs,already_embedded=True)#cbfd(znext_obs_mean)
                    hdiff=hnextobs-hobs
                    hdiffnorm=np.linalg.norm(hdiff)
                    slopexyp=imagediffnormal/(posdiffnorm+eps)
                    slopeyzp=zdiffnorm/(imagediffnormal+eps)
                    slopezhp=hdiffnorm/(zdiff+eps)
                    slopeyhp=hdiffnorm/(imagediffnormal+eps)
                    slopexy[piece]=slopexyp
                    slopeyz[piece]=slopeyzp
                    slopezh[piece]=slopezhp
                    slopeyh[piece]=slopeyhp







                    obs = next_obs#don't forget this step!
                    #print('obs.shape',obs.shape)#(3, 3, 64, 64)
                    #obs_relative = next_obs_relative  # don't forget this step!
                    oldcviol=constr_viol
                    constr_viol = constr_viol or info['constraint']#a way to update constr_viol#either 0 or 1
                    constr_viol_cbf = constr_viol_cbf or constr_cbf#a way to update constr_viol#either 0 or 1
                    constr_viol_cbf2 = constr_viol_cbf2 or constr_cbf2#a way to update constr_viol#either 0 or 1
                    succ = succ or reward == 0#as said in the paper, reward=0 means success!

                    '''
                    #Now, I should do the evaluation!
                    obseval= ptu.torchify(obs).reshape(1, *obs.shape)#it seems that this reshaping is necessary
                    #obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing#pay attention to its shape!#prepare to be used!
                    #embeval = encoder.encode(obseval)#in latent space now!
                    if params['mean']=='sample':
                        embeval = encoder.encode(obseval)#in latent space now!#even
                    elif params['mean']=='mean':
                        embeval = encoder.encodemean(obseval)#in latent space now!#really zero now! That's what I  want!
                        #embeval2 = encoder.encodemean(obseval)#in latent space now!
                    #print('emb.shape',emb.shape)#torch.Size([1, 32])
                    #cbfdot_function.predict()
                    
                    cbfpredict = cbfdot_function(embeval,already_embedded=True)#is it legit? Seems not!
                    cbfgt=hvn
                    if (cbfpredict>=0) and (cbfgt>=0):
                        tn+=1
                    elif (cbfpredict>=0) and (cbfgt<0):
                        fn+=1
                    elif (cbfpredict<0) and (cbfgt>=0):
                        fp+=1
                    elif (cbfpredict<0) and (cbfgt<0):
                        tp+=1
                    tncvalue=0.05**2-0.055**2+1e-4#for reacher!#0.05**2-0.06**2+1e-4#0.3**2-0.4**2+1e-3#FOR PUSHING!#
                    if (cbfpredict>=0) and (cbfgt>=tncvalue):
                        tnc+=1
                    elif (cbfpredict>=0) and (cbfgt<tncvalue):
                        fnc+=1
                    elif (cbfpredict<0) and (cbfgt>=tncvalue):
                        fpc+=1
                    elif (cbfpredict<0) and (cbfgt<tncvalue):
                        tpc+=1
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,s_x:%f,s_y:%f,c_viol:%d,c_viol_cbf:%d,c_viol_cbf2:%d,a_rand:%d' % (tp, fp, fn, tn, tpc, fpc, fnc, tnc,ns[0],ns[1],constr_viol,constr_viol_cbf,constr_viol_cbf2,action_rand))
                    '''
                    log.info('s_x:%f,s_y:%f,c_viol:%d,c_viol_cbf:%d,c_viol_cbf2:%d,a_rand:%d' % (ns[0],ns[1],constr_viol,constr_viol_cbf,constr_viol_cbf2,action_rand))



                    




                    #the evaluation phase ended
                    #if done:#when calculating lipschitz constant, I want it to be 500 steps, so disable this part
                        #break
                    piece+=1
                    if (oldcviol and constr_viol)==1:#one step buffer/hold
                        break
                transitions[-1]['done'] = 1#change the last transition to success/done!
                traj_reward = sum(traj_rews)#total reward, should be >=-100/-150
                #EpRet is episode reward, EpLen=Episode Length, EpConstr=Episode constraints
                logger.store(EpRet=traj_reward, EpLen=k+1, EpConstr=float(constr_viol))
                all_rewards.append(traj_rews)#does it use any EpLen?
                all_action_rands.append(traj_action_rands)
                constr_viols.append(constr_viol)#whether this 100-length traj violate any constraints, then compute the average
                constr_viols_cbf.append(constr_viol_cbf)#
                constr_viols_cbf2.append(constr_viol_cbf2)#
                task_succ.append(succ)
                #save the result in the gift form!
                pu.make_movie(movie_traj, file=os.path.join(update_dir, 'trajectory%d.gif' % j))
                #pu.make_movie_relative(movie_traj_relative, file=os.path.join(update_dir, 'trajectory%d_relative.gif' % j))

                #log.info('    Cost: %d, constraint violation: %d' % (traj_reward,constr_viol))#see it in the terminal!
                log.info('    Cost: %d, constraint violation: %d, cv_cbf: %d, cv_cbf2: %d' % (traj_reward,constr_viol,constr_viol_cbf,constr_viol_cbf2))#see it in the terminal!
                #log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d' % (tp, fp, fn, tn, tpc, fpc, fnc, tnc))
                in_ss = 0
                rtg = 0
                for transition in reversed(transitions):
                    if transition['reward'] > -1:
                        in_ss = 1
                    transition['safe_set'] = in_ss
                    transition['rtg'] = rtg

                    rtg = rtg + transition['reward']

                replay_buffer.store_transitions(transitions)#replay buffer online training
                #I am going to save trajectory!
                #utils.save_trajectory(traj, file, i)#
                utils.save_trajectory(transitions, datasave_dir, j)#
                update_rewards.append(traj_reward)

            mean_rew = float(np.mean(update_rewards))
            std_rew = float(np.std(update_rewards))
            avg_rewards.append(mean_rew)
            std_rewards.append(std_rew)
            log.info('Iteration %d average reward: %.4f' % (i, mean_rew))
            pu.simple_plot(avg_rewards, std=std_rewards, title='Average Rewards',
                        file=os.path.join(logdir, 'rewards.pdf'),
                        ylabel='Average Reward', xlabel='# Training updates')

            logger.log_tabular('Epoch', i)
            logger.log_tabular('TrainEpisodes', n_episodes)
            logger.log_tabular('TestEpisodes', traj_per_update)
            logger.log_tabular('EpRet')
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('EpConstr', average_only=True)
            logger.log_tabular('ConstrRate', np.mean(constr_viols))
            logger.log_tabular('ConstrcbfRate', np.mean(constr_viols_cbf))
            logger.log_tabular('Constrcbf2Rate', np.mean(constr_viols_cbf2))
            logger.log_tabular('SuccRate', np.mean(task_succ))
            logger.dump_tabular()
            n_episodes += traj_per_update#10 by default

            # Update models

            #episodiccbfdhz=trainer.update(replay_buffer, i,replay_buffer_unsafe)#online training, right?#not needed when calculating the Lipschitz of a CBF
            #if params['dynamic_dhz']=='yes':
                #dhzoriginal=params['dhz']
                #log.info('old dhz: %f'%(dhzoriginal))#not needed, as it is already printed at the begining of each episode
                #params['dhz']=(1-cbfalpha)*dhzoriginal+cbfalpha*episodiccbfdhz
            log.info('new dhz: %f'%(params['dhz']))#if dynamic_dhz=='no', then it will be still the old dhz
            np.save(os.path.join(logdir, 'rewards.npy'), all_rewards)
            np.save(os.path.join(logdir, 'constr.npy'), constr_viols)
            np.save(os.path.join(logdir, 'constrcbf.npy'), constr_viols_cbf)
            np.save(os.path.join(logdir, 'constrcbf2.npy'), constr_viols_cbf2)
            np.save(os.path.join(logdir, 'action_rands.npy'), all_action_rands)
            np.save(os.path.join(logdir, 'tasksuccess.npy'), task_succ)
        params['seed']=params['seed']+1#m+1#
        #utils.init_logging(logdir)#record started!
        #logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M:%S',filename=os.path.join(logdir, 'logjianning.txt'),filemode='w')