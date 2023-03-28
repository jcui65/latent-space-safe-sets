#data processing helper
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/cuijin/Project6remote/latent-space-safe-sets')

import os
import numpy as np
from latentsafesets.utils.arg_parser import parse_args
import latentsafesets.utils.plot_utils as pu
#load data from the corresponding folder
#params = parse_args()#get the parameters from parse_args, see arg_parser.py
outputdir='/home/cuijin/Project6remote/latent-space-safe-sets/outputs/'
mar24='2023-03-24'
mar23='2023-03-23'
mar22='2023-03-22'
mar25='2023-03-25'
mar27='2023-03-27'
date=mar25#mar23#mar24#mar22#
time='20-05-38'#'20-07-08'#'09-30-18'#'13-24-32'#'19-45-12'#'14-02-35'#'17-06-29'#'11-53-02'#'11-51-10'#'11-34-01'#'11-33-24'#'14-03-38'#'13-24-46'#'00-54-16'#'19-45-47'#
srlist=[]#success rate list
ralist=[]#reward average list
rfarray=np.zeros((250,))
ralastlist=[]#reward average last list
cvrlist=[]#constraint violation rate list
lastnum=50
logdirbeforeseed = os.path.join(outputdir+date,time) #params['logdir']#around line 35
#logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
print('logdirbeforeseed',logdirbeforeseed)
seedlist=[1,2,3]#[1,101,201]#23#[4,5,6,7,8,9,10]#22#[1,26,51]#24#
for seed in seedlist:
    logdir=os.path.join(logdirbeforeseed, str(seed))
    #update_dir = os.path.join(logdir, "update_%d" % i)#
    rewardi=np.load(os.path.join(logdir, "rewards.npy"))
    #print('rewardi.shape',rewardi.shape)#250,100
    rewardsumi=np.sum(rewardi,axis=1)
    #print('rewardsumi.shape',rewardsumi.shape)#250
    rfarray=np.vstack((rfarray,rewardsumi))
    successi=rewardsumi>-100
    successratei=np.average(successi)
    #print('successrate',successratei)
    rewardaveragei=np.average(rewardsumi)
    print('rewardaveragei',rewardaveragei)
    #print('shape',rewardsumi[-lastnum:].shape)#it is 50
    rewardaveragelasti=np.sum(rewardsumi[-lastnum:])/lastnum
    print('rewardaveragelasti',rewardaveragelasti)
    constri=np.load(os.path.join(logdir, "constr.npy"))
    #print('constri.shape',constri.shape)#250
    totalconstri=np.sum(constri)
    constrirate=totalconstri/constri.shape[0]
    #print('constrirate',constrirate)
    srlist.append(successratei)
    cvrlist.append(constrirate)
    ralist.append(rewardaveragei)
    ralastlist.append(rewardaveragelasti)

date=mar27#mar25#mar23#mar22#mar24#
time='00-08-25'#'00-04-54'#'20-22-45'#'20-22-06'#'17-06-29'#'17-27-44'#'18-37-20'#'15-53-12'#
logdirbeforeseed = os.path.join(outputdir+date,time) #params['logdir']#around line 35
#logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
print('logdirbeforeseed',logdirbeforeseed)
seedlist=[4,5,6,7,8,9,10]#[1,101,201]#22#[1,26,51]#[1,2,3]#24#23#
for seed in seedlist:
    logdir=os.path.join(logdirbeforeseed, str(seed))
    #update_dir = os.path.join(logdir, "update_%d" % i)#
    rewardi=np.load(os.path.join(logdir, "rewards.npy"))
    #print('rewardi.shape',rewardi.shape)#250,100
    rewardsumi=np.sum(rewardi,axis=1)
    #print('rewardsumi.shape',rewardsumi.shape)#250
    rfarray=np.vstack((rfarray,rewardsumi))
    successi=rewardsumi>-100
    successratei=np.average(successi)
    #print('successrate',successratei)
    rewardaveragei=np.average(rewardsumi)
    print('rewardaveragei',rewardaveragei)
    #print('shape',rewardsumi[-lastnum:].shape)#it is 50
    rewardaveragelasti=np.sum(rewardsumi[-lastnum:])/lastnum
    print('rewardaveragelasti',rewardaveragelasti)
    constri=np.load(os.path.join(logdir, "constr.npy"))
    #print('constri.shape',constri.shape)#250
    totalconstri=np.sum(constri)
    constrirate=totalconstri/constri.shape[0]
    #print('constrirate',constrirate)
    srlist.append(successratei)
    cvrlist.append(constrirate)
    ralist.append(rewardaveragei)
    ralastlist.append(rewardaveragelasti)

#calculate the statistics: mean and std
rfarray=rfarray[1:]
#print('rfarray.shape',rfarray.shape)#3,250
rfmean=np.mean(rfarray,axis=0)
#print(rfmean)
rfstd=np.std(rfarray,axis=0)
#print(rfstd)
pu.simple_plot(rfmean, std=rfstd, title='Average Rewards',
                        file=os.path.join(logdirbeforeseed, 'rewards10trajs'+date+time+'.pdf'),
                        ylabel='Average Reward', xlabel='# Training updates')
rfcarray=np.zeros((10,))#np.zeros((len(seedlist),))#c means corse#
for i in range(int(rfarray.shape[1]/10)):
    rfciclip=rfarray[:,i*10:(i+1)*10]
    #print(rfciclip)
    rfci=np.average(rfciclip,axis=1)
    #print('rfci.shape',rfci)
    rfcarray=np.vstack((rfcarray,rfci))
rfcarray=rfcarray[1:]
#print('rfcarray.shape',rfcarray.shape)#3,250
rfcmean=np.mean(rfcarray,axis=1)
#print(rfcmean)
rfcstd=np.std(rfcarray,axis=1)
#print(rfcstd)
pu.simple_plot(rfcmean, std=rfcstd, title='Average Rewards',
                        file=os.path.join(logdirbeforeseed, 'rewards10epochs'+date+time+'.pdf'),
                        ylabel='Average Reward', xlabel='# Training updates')
sra=np.array(srlist)
cvra=np.array(cvrlist)
raa=np.array(ralist)
ralasta=np.array(ralastlist)
sraave=np.mean(sra)
srastd=np.std(sra)
print('successrate ave',sraave,'successrate std',srastd)
pu.simple_plot(sra, title='Success rate %f'%(sraave)+"\u00B1"+'%f'%(srastd),
                        file=os.path.join(logdirbeforeseed, 'success10rate'+date+time+'.pdf'),
                        ylabel='success rate', xlabel='# seeds',nonreward=True)
cvraave=np.mean(cvra)
cvrastd=np.std(cvra)
print('constraint rate ave',cvraave,'constraint rate std',cvrastd)
pu.simple_plot(cvra, title='Constraint violation rate %f'%(cvraave)+"\u00B1"+'%f'%(cvrastd),
                        file=os.path.join(logdirbeforeseed, 'violation10rate'+date+time+'.pdf'),
                        ylabel='constraint violation rate', xlabel='# seeds',nonreward=True)
print('reward ave',np.mean(raa),'reward std',np.std(raa))
print('reward last ave',np.mean(ralasta),'reward last std',np.std(ralasta))
#making plots