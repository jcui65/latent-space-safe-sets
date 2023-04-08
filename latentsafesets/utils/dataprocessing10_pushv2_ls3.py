#data processing helper
import sys
import click
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/cuijin/Project6remote/latent-space-safe-sets')

import os
import numpy as np
from latentsafesets.utils.arg_parser import parse_args
import latentsafesets.utils.plot_utils as pu
#load data from the corresponding folder
#params = parse_args()#get the parameters from parse_args, see arg_parser.py
@click.command()
@click.option('--dateo', default='04-01',help='the date1 when the simulation started', type=str)
@click.option('--timeo', default='14-11-42', help='time1 of the simulation', type=str)
@click.option('--datet', default='04-04',help='the date2 when the simulation started', type=str)
@click.option('--timet', default='13-25-17', help='time2 of the simulation', type=str)
@click.option('--dateth', default='04-04',help='the date3 when the simulation started', type=str)
@click.option('--timeth', default='13-28-34', help='time3 of the simulation', type=str)
@click.option('--datef', default='04-04',help='the date4 when the simulation started', type=str)
@click.option('--timef', default='13-27-02', help='time4 of the simulation', type=str)
def main(dateo, timeo,datet, timet,dateth, timeth,datef, timef):
    outputdir='/home/cuijin/Project6remote/latent-space-safe-sets/outputs/2023-'
    #date=mar28#mar26#mar27#mar25#mar23#mar22#mar24#
    #time='20-29-18'#'20-28-35'#'20-26-13'#'20-23-19'#'00-08-25'#'00-04-54'#'00-02-15'#'20-22-45'#'20-22-06'#'20-11-00'#'20-09-26'#'20-07-08'#'20-05-38'#'19-38-18'#'15-36-29'#'15-35-54'#'15-22-41'#'14-54-22'#'14-53-10'#'18-37-20'#'17-27-44'#'17-06-29'#
    #time='15-53-12'#'09-30-18'##'11-53-02'#'11-51-10'#'11-34-01'#'11-33-24'#'13-24-32'#'19-45-12'#'14-02-35'#'14-03-38'#'13-24-46'#'00-54-16'#'19-45-47'#
    
    srlist=[]#success rate list
    ralist=[]#reward average list
    fh=500#1000#fh means five hundred
    rfarray=np.zeros((fh,))#np.zeros((1000,))#push#np.zeros((250,))#reacher
    tsrarray=np.zeros((fh,))#np.zeros((1000,))#reacher#np.zeros((1000,))#push#
    ralastlist=[]#reward average last list
    cvrlist=[]#constraint violation rate list
    cvrcbflist=[]#constraint violation rate list
    cvrcbf2list=[]#constraint violation rate list
    lastnum=50
    seedlist=[1,2,3]#24,25#[4,5,6,7,8,9,10]#[1,101,201]#23#22#[1,26,51]##
    logdirbeforeseed = os.path.join(outputdir+dateo,timeo) #params['logdir']#around line 35
    #logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
    print('logdirbeforeseed',logdirbeforeseed)
    for seed in seedlist:
        logdir=os.path.join(logdirbeforeseed, str(seed))
        #update_dir = os.path.join(logdir, "update_%d" % i)#
        rewardi=np.load(os.path.join(logdir, "rewards.npy"))
        #print('rewardi.shape',rewardi.shape)#250,100
        rewardsumi=np.sum(rewardi[0:fh],axis=1)
        #print('rewardsumi.shape',rewardsumi.shape)#500
        rfarray=np.vstack((rfarray,rewardsumi))
        successi=rewardsumi>-150#push#-100#reacher
        tsrarray=np.vstack((tsrarray,successi))
        successratei=np.average(successi)
        #print('successrate',successratei)
        rewardaveragei=np.average(rewardsumi)
        print('rewardaveragei',rewardaveragei)
        #print('shape',rewardsumi[-lastnum:].shape)#it is 50
        rewardaveragelasti=np.sum(rewardsumi[fh-lastnum:fh])/lastnum
        print('rewardaveragelasti',rewardaveragelasti)
        constri=np.load(os.path.join(logdir, "constr.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstri=np.sum(constri[0:fh])
        constrirate=totalconstri/fh#constri.shape[0]
        constrcbfi=np.load(os.path.join(logdir, "constrcbf.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstrcbfi=np.sum(constrcbfi[0:fh])
        constrcbfirate=totalconstrcbfi/fh#constrcbfi.shape[0]
        constrcbf2i=np.load(os.path.join(logdir, "constrcbf2.npy"))
        totalconstrcbf2i=np.sum(constrcbf2i[0:fh])
        constrcbf2irate=totalconstrcbf2i/fh#constrcbf2i.shape[0]
        #print('constrirate',constrirate)
        srlist.append(successratei)
        cvrlist.append(constrirate)
        cvrcbflist.append(constrcbfirate)
        cvrcbf2list.append(constrcbf2irate)
        ralist.append(rewardaveragei)
        ralastlist.append(rewardaveragelasti)
    seedlist2=[4,5]#
    logdirbeforeseed2 = os.path.join(outputdir+datet,timet) #params['logdir']#around line 35
    #logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
    print('logdirbeforeseed2',logdirbeforeseed2)
    for seed in seedlist2:
        logdir=os.path.join(logdirbeforeseed2, str(seed))
        #update_dir = os.path.join(logdir, "update_%d" % i)#
        rewardi=np.load(os.path.join(logdir, "rewards.npy"))
        #print('rewardi.shape',rewardi.shape)#250,100
        rewardsumi=np.sum(rewardi[0:fh],axis=1)
        #print('rewardsumi.shape',rewardsumi.shape)#500
        rfarray=np.vstack((rfarray,rewardsumi))
        successi=rewardsumi>-150#push#-100#reacher
        tsrarray=np.vstack((tsrarray,successi))
        successratei=np.average(successi)
        #print('successrate',successratei)
        rewardaveragei=np.average(rewardsumi)
        print('rewardaveragei',rewardaveragei)
        #print('shape',rewardsumi[-lastnum:].shape)#it is 50
        rewardaveragelasti=np.sum(rewardsumi[fh-lastnum:fh])/lastnum
        print('rewardaveragelasti',rewardaveragelasti)
        constri=np.load(os.path.join(logdir, "constr.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstri=np.sum(constri[0:fh])
        constrirate=totalconstri/fh#constri.shape[0]
        constrcbfi=np.load(os.path.join(logdir, "constrcbf.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstrcbfi=np.sum(constrcbfi[0:fh])
        constrcbfirate=totalconstrcbfi/fh#constrcbfi.shape[0]
        constrcbf2i=np.load(os.path.join(logdir, "constrcbf2.npy"))
        totalconstrcbf2i=np.sum(constrcbf2i[0:fh])
        constrcbf2irate=totalconstrcbf2i/fh#constrcbf2i.shape[0]
        #print('constrirate',constrirate)
        srlist.append(successratei)
        cvrlist.append(constrirate)
        cvrcbflist.append(constrcbfirate)
        cvrcbf2list.append(constrcbf2irate)
        ralist.append(rewardaveragei)
        ralastlist.append(rewardaveragelasti)
    seedlist3=[6,7]#
    logdirbeforeseed3 = os.path.join(outputdir+dateth,timeth) #params['logdir']#around line 35
    #logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
    print('logdirbeforeseed3',logdirbeforeseed3)
    for seed in seedlist3:
        logdir=os.path.join(logdirbeforeseed3, str(seed))
        #update_dir = os.path.join(logdir, "update_%d" % i)#
        rewardi=np.load(os.path.join(logdir, "rewards.npy"))
        #print('rewardi.shape',rewardi.shape)#250,100
        rewardsumi=np.sum(rewardi[0:fh],axis=1)
        #print('rewardsumi.shape',rewardsumi.shape)#500
        rfarray=np.vstack((rfarray,rewardsumi))
        successi=rewardsumi>-150#push#-100#reacher
        tsrarray=np.vstack((tsrarray,successi))
        successratei=np.average(successi)
        #print('successrate',successratei)
        rewardaveragei=np.average(rewardsumi)
        print('rewardaveragei',rewardaveragei)
        #print('shape',rewardsumi[-lastnum:].shape)#it is 50
        rewardaveragelasti=np.sum(rewardsumi[fh-lastnum:fh])/lastnum
        print('rewardaveragelasti',rewardaveragelasti)
        constri=np.load(os.path.join(logdir, "constr.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstri=np.sum(constri[0:fh])
        constrirate=totalconstri/fh#constri.shape[0]
        constrcbfi=np.load(os.path.join(logdir, "constrcbf.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstrcbfi=np.sum(constrcbfi[0:fh])
        constrcbfirate=totalconstrcbfi/fh#constrcbfi.shape[0]
        constrcbf2i=np.load(os.path.join(logdir, "constrcbf2.npy"))
        totalconstrcbf2i=np.sum(constrcbf2i[0:fh])
        constrcbf2irate=totalconstrcbf2i/fh#constrcbf2i.shape[0]
        #print('constrirate',constrirate)
        srlist.append(successratei)
        cvrlist.append(constrirate)
        cvrcbflist.append(constrcbfirate)
        cvrcbf2list.append(constrcbf2irate)
        ralist.append(rewardaveragei)
        ralastlist.append(rewardaveragelasti)
    seedlist4=[8,9,10]#
    logdirbeforeseed4 = os.path.join(outputdir+datef,timef) #params['logdir']#around line 35
    #logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
    print('logdirbeforeseed4',logdirbeforeseed4)
    for seed in seedlist4:
        logdir=os.path.join(logdirbeforeseed4, str(seed))
        #update_dir = os.path.join(logdir, "update_%d" % i)#
        rewardi=np.load(os.path.join(logdir, "rewards.npy"))
        #print('rewardi.shape',rewardi.shape)#250,100
        rewardsumi=np.sum(rewardi[0:fh],axis=1)
        #print('rewardsumi.shape',rewardsumi.shape)#500
        rfarray=np.vstack((rfarray,rewardsumi))
        successi=rewardsumi>-150#push#-100#reacher
        tsrarray=np.vstack((tsrarray,successi))
        successratei=np.average(successi)
        #print('successrate',successratei)
        rewardaveragei=np.average(rewardsumi)
        print('rewardaveragei',rewardaveragei)
        #print('shape',rewardsumi[-lastnum:].shape)#it is 50
        rewardaveragelasti=np.sum(rewardsumi[fh-lastnum:fh])/lastnum
        print('rewardaveragelasti',rewardaveragelasti)
        constri=np.load(os.path.join(logdir, "constr.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstri=np.sum(constri[0:fh])
        constrirate=totalconstri/fh#constri.shape[0]
        constrcbfi=np.load(os.path.join(logdir, "constrcbf.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstrcbfi=np.sum(constrcbfi[0:fh])
        constrcbfirate=totalconstrcbfi/fh#constrcbfi.shape[0]
        constrcbf2i=np.load(os.path.join(logdir, "constrcbf2.npy"))
        totalconstrcbf2i=np.sum(constrcbf2i[0:fh])
        constrcbf2irate=totalconstrcbf2i/fh#constrcbf2i.shape[0]
        #print('constrirate',constrirate)
        srlist.append(successratei)
        cvrlist.append(constrirate)
        cvrcbflist.append(constrcbfirate)
        cvrcbf2list.append(constrcbf2irate)
        ralist.append(rewardaveragei)
        ralastlist.append(rewardaveragelasti)
    
    #calculate the statistics: mean and std
    rfarray=rfarray[1:]
    tsrarray=tsrarray[1:]
    #print('rfarray.shape',rfarray.shape)#3,250
    rfmean=np.mean(rfarray,axis=0)
    tsrmean=np.mean(tsrarray,axis=0)
    #print(rfmean)
    rfstd=np.std(rfarray,axis=0)
    tsrstd=np.std(tsrarray,axis=0)
    #print(rfstd)
    lenseed=10#len(seedlist)
    pu.simple_plot(rfmean, std=rfstd, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed4, 'rewards'+str(lenseed)+'trajs'+datef+'-'+timef+'ls3'+'epochs'+str(fh)+'.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot(tsrmean, std=tsrstd, title='Average task success rate',
                            file=os.path.join(logdirbeforeseed4, 'tsr'+str(lenseed)+'trajs'+datef+'-'+timef+'ls3'+'epochs'+str(fh)+'.pdf'),
                            ylabel='Average task success rate', xlabel='# Training updates')
    rfcarray=np.zeros((lenseed,))#c means corse
    tsrcarray=np.zeros((lenseed,))#c means corse
    for i in range(int(rfarray.shape[1]/10)):
        rfciclip=rfarray[:,i*10:(i+1)*10]
        tsrciclip=tsrarray[:,i*10:(i+1)*10]
        #print(rfciclip)
        rfci=np.average(rfciclip,axis=1)
        tsrci=np.average(tsrciclip,axis=1)
        #print('rfci.shape',rfci)
        rfcarray=np.vstack((rfcarray,rfci))
        tsrcarray=np.vstack((tsrcarray,tsrci))
    rfcarray=rfcarray[1:]
    tsrcarray=tsrcarray[1:]
    #print('rfcarray.shape',rfcarray.shape)#3,250
    rfcmean=np.mean(rfcarray,axis=1)
    tsrcmean=np.mean(tsrcarray,axis=1)
    #print(rfcmean)
    rfcstd=np.std(rfcarray,axis=1)
    tsrcstd=np.std(tsrcarray,axis=1)
    #print(rfcstd)
    pu.simple_plot(rfcmean, std=rfcstd, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed4, 'rewards'+str(lenseed)+'epochs'+datef+'-'+timef+'ls3'+'epochs'+str(fh)+'.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot(tsrcmean, std=tsrcstd, title='Average task success rate',
                            file=os.path.join(logdirbeforeseed4, 'tsrc'+str(lenseed)+'epochs'+datef+'-'+timef+'ls3'+'epochs'+str(fh)+'.pdf'),
                            ylabel='Average  task success rate', xlabel='# Training updates')
    sra=np.array(srlist)
    cvra=np.array(cvrlist)
    cvrcbfa=np.array(cvrcbflist)
    cvrcbf2a=np.array(cvrcbf2list)
    raa=np.array(ralist)
    ralasta=np.array(ralastlist)
    sraave=np.mean(sra)
    srastd=np.std(sra)
    print('successrate ave',sraave,'successrate std',srastd)
    pu.simple_plot(sra, title='Success rate %f'%(sraave)+"\u00B1"+'%f'%(srastd),
                            file=os.path.join(logdirbeforeseed4, 'success'+str(lenseed)+'rate'+datef+'-'+timef+'ls3'+'epochs'+str(fh)+'.pdf'),
                            ylabel='success rate', xlabel='# seeds',nonreward=True)
    cvraave=np.mean(cvra)
    cvrastd=np.std(cvra)
    print('constraint rate ave',cvraave,'constraint rate std',cvrastd)
    pu.simple_plot(cvra, title='Constraint violation rate %f'%(cvraave)+"\u00B1"+'%f'%(cvrastd),
                            file=os.path.join(logdirbeforeseed4, 'violation'+str(lenseed)+'rate'+datef+'-'+timef+'ls3'+'epochs'+str(fh)+'.pdf'),
                            ylabel='constraint violation rate', xlabel='# seeds',nonreward=True)
    cvrcbfaave=np.mean(cvrcbfa)
    cvrcbfastd=np.std(cvrcbfa)
    print('constraint rate cbf ave',cvrcbfaave,'constraint rate cbf std',cvrcbfastd)
    pu.simple_plot(cvrcbfa, title='Constraint violation cbf rate %f'%(cvrcbfaave)+"\u00B1"+'%f'%(cvrcbfastd),
                            file=os.path.join(logdirbeforeseed4, 'violation'+str(lenseed)+'rate'+datef+'-'+timef+'safetyls3'+'epochs'+str(fh)+'.pdf'),
                            ylabel='constraint violation cbf rate', xlabel='# seeds',nonreward=True)
    cvrcbf2aave=np.mean(cvrcbf2a)
    cvrcbf2astd=np.std(cvrcbf2a)
    print('constraint rate cbf2 ave',cvrcbf2aave,'constraint rate cbf2 std',cvrcbfastd)
    pu.simple_plot(cvrcbf2a, title='Constraint violation cbf2 rate %f'%(cvrcbf2aave)+"\u00B1"+'%f'%(cvrcbf2astd),
                            file=os.path.join(logdirbeforeseed4, 'violation'+str(lenseed)+'rate'+datef+'-'+timef+'safety2ls3'+'epochs'+str(fh)+'.pdf'),
                            ylabel='constraint violation cbf2 rate', xlabel='# seeds',nonreward=True)
    print('reward ave',np.mean(raa),'reward std',np.std(raa))
    print('reward last ave',np.mean(ralasta),'reward last std',np.std(ralasta))
    #making plots

if __name__=='__main__':
    main()