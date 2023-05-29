
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


if __name__ == '__main__':
    params = parse_args()#get the parameters from parse_args, see arg_parser.py
    # Misc preliminaries
    repeattimes=params['repeat_times']
    initdhz=params['dhz']
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
        f.write('nosigma: %f, nosigmadhz: %f, dynamic_dhz: %s, idea: %s, unsafe_buffer:%s, reg_lipschitz:%s\n'%(params['noofsigma'],params['noofsigmadhz'],params['dynamic_dhz'],params['idea'],params['unsafebuffer'],params['reg_lipschitz']))
        f.write('What I want to write in m2: %s'%(params['quote']))
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
        replay_buffer_success = utils.load_replay_buffer_success(params, encoder)#around line 123 in utils.py
        #if reacher:
        #then also load random interactions
        #replay_buffer = utils.load_replay_buffer_relative(params, encoder)  # around line 123 in utils.py
        replay_buffer_unsafe = utils.load_replay_buffer_unsafe(params, encoder)#around line 123 in utils.py
        log.info('unsafe buffer!')

        trainer = MPCTrainer(env, params, modules)#so that we can train MPC!

        #trainer.initial_train(replay_buffer_success,replay_buffer_unsafe)#initialize all the parts!
        trainer.initial_train_m2(replay_buffer_success,replay_buffer_unsafe)#initialize all the parts!

        log.info("Creating policy")
        #policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,
                                #constraint_function, goal_indicator, params)
        policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,
                                constraint_function, goal_indicator, cbfdot_function, params)

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
        action_type=params['action_type']#can be deleted, not used anymore!!
        cbfalpha=0.2#exponential averaging for CBF
        for i in range(num_updates):#default 25 in spb
            log.info('current dhz: %f'%(params['dhz']))
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
                constr_viol = False
                succ = False
                traj_action_rands=[]
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
                    
                    #action,randflag= policy.actcbfdsquarelatentplanareacher(obs / 255)#,conservative,reward_type)#
                    action,randflag= policy.actcbfdsquarelatentplanareacher(obs / 255,params['dhz'])#
                    '''
                    if conservative=='conservative' and reward_type=='sparse':
                        #print('conservative and sparse!')#you get this right!
                        action,randflag= policy.actcbfdsquarelatentplanareacher(obs / 255)#, env.state)#, tp, fp,#obs_relative / 255, env.state, tp, fp,#
                                                                                            #fn, tn, tpc, fpc, fnc, tnc)
                    elif conservative=='average' and reward_type=='sparse':
                        action,randflag= policy.actcbfdsquarelatentplanareacheraverage(obs / 255)#, env.state)#
                    elif conservative=='onestd' and reward_type=='sparse':
                        action,randflag= policy.actcbfdsquarelatentplanareacheronestd(obs / 255)#, env.state)#
                    elif conservative=='conservative' and reward_type=='dense':
                        action,randflag= policy.actcbfdsquarelatentplanareachernogoaldense(obs / 255)#, env.state)#
                    elif conservative=='average' and reward_type=='dense':
                        action,randflag= policy.actcbfdsquarelatentplanareacheraveragenogoaldense(obs / 255)#, env.state)#
                    elif conservative=='average' and reward_type=='dense':
                        action,randflag= policy.actcbfdsquarelatentplanareacheronestdnogoaldense(obs / 255)#, env.state)#
                    '''
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
                    if params['mean']=='sample':
                        embeval = encoder.encode(obseval/255)#encoder.encode(obseval)#in latent space now!#even
                        #obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing#pay attention to its shape!#prepare to be used!
                        #embeval2 = encoder.encode(obseval)#in latent space now!
                    elif params['mean']=='mean' or params['mean']=='meancbf':
                        embeval = encoder.encodemean(obseval/255)#encoder.encodemean(obseval)#in latent space now!#really zero now! That's what I  want!
                        #embeval2 = encoder.encodemean(obseval)#in latent space now!
                    #embdiff100000=(embeval-embeval2)*100000
                    #print('embdiff100000',embdiff100000)#just for testing!!!
                    #embdiffmax,ind=torch.max(embdiff)
                    #print('embdiffmax',embdiffmax)
                    #print('emb.shape',emb.shape)#torch.Size([1, 32])
                    #cbfdot_function.predict()
                    cbfpredict = cbfdot_function(embeval,already_embedded=True)#
                    cbfgt=hvn
                    if (cbfpredict>=0) and (cbfgt>=0):
                        tn+=1
                    elif (cbfpredict>=0) and (cbfgt<0):
                        fn+=1
                    elif (cbfpredict<0) and (cbfgt>=0):
                        fp+=1
                    elif (cbfpredict<0) and (cbfgt<0):
                        tp+=1
                    tncvalue=0.05**2-0.06**2+1e-4#for reacher!#0.05**2-0.06**2+1e-4#0.3**2-0.4**2+1e-3#FOR PUSHING!#
                    if (cbfpredict>=0) and (cbfgt>=tncvalue):
                        tnc+=1
                    elif (cbfpredict>=0) and (cbfgt<tncvalue):
                        fnc+=1
                    elif (cbfpredict<0) and (cbfgt>=tncvalue):
                        fpc+=1
                    elif (cbfpredict<0) and (cbfgt<tncvalue):
                        tpc+=1
                    log.info('tp:%d,fp:%d,fn:%d,tn:%d,tpc:%d,fpc:%d,fnc:%d,tnc:%d,s_x:%f,s_y:%f,c_viol:%d,c_viol_cbf:%d,c_viol_cbf2:%d,a_rand:%d' % (tp, fp, fn, tn, tpc, fpc, fnc, tnc,ns[0],ns[1],constr_viol,constr_viol_cbf,constr_viol_cbf2,action_rand))
                    
                    #the evaluation phase ended
                    if done:
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

            episodiccbfdhz=trainer.update(replay_buffer_success, i,replay_buffer_unsafe)#online training, right?#it now only bears the meaning of dhz!
            if params['dynamic_dhz']=='yes':
                dhzoriginal=params['dhz']
                #log.info('old dhz: %f'%(dhzoriginal))#not needed, as it is already printed at the begining of each episode
                params['dhz']=(1-cbfalpha)*dhzoriginal+cbfalpha*episodiccbfdhz*params['noofsigmadhz']*(2-params['cbfdot_thresh'])
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