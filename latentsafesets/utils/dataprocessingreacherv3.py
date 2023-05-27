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
@click.option('--date1', default='05-01',help='the date when the simulation started', type=str)
@click.option('--time1', default='01-19-14', help='time of the simulation', type=str)
@click.option('--date2', default='05-16',help='the date when the simulation started', type=str)
@click.option('--time2', default='23-10-18', help='time of the simulation', type=str)
@click.option('--fh', default=250, help='five hundred or 250', type=int)
def main(date1, time1,date2, time2,fh):
    outputdir='/home/cuijin/Project6remote/latent-space-safe-sets/outputs/2023-'
    logdirbeforeseed = os.path.join(outputdir+date1,time1) #params['logdir']#around line 35
    #logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
    print('logdirbeforeseed',logdirbeforeseed)
    srlist=[]#success rate list
    ralist=[]#reward average list
    #fh=500#250#fh means 500
    rfarray=np.zeros((fh,))#reacher#np.zeros((1000,))#push#
    cvarray=np.zeros((fh,))#reacher#np.zeros((1000,))#push#
    ralastlist=[]#reward average last list
    cvrlist=[]#constraint violation rate list
    cvrcbflist=[]#constraint violation rate list
    cvrcbf2list=[]#constraint violation rate list
    tsrarray=np.zeros((fh,))#reacher#np.zeros((1000,))#push#
    seedlist=[1,2,3]#[1,2]#[1,2,3,4,5]#24,25#[1,2,3,4,5,6,7,8,9,10]#[1,101,201]#22#[4,5,6,7,8,9,10]#23#[1,26,51]##

    for seed in seedlist:
        logdir=os.path.join(logdirbeforeseed, str(seed))
        #update_dir = os.path.join(logdir, "update_%d" % i)#
        rewardi=np.load(os.path.join(logdir, "rewards.npy"))
        #print('rewardi.shape',rewardi.shape)#250,100
        rewardsumi=np.sum(rewardi[0:fh],axis=1)
        #print('rewardsumi.shape',rewardsumi.shape)#250
        rfarray=np.vstack((rfarray,rewardsumi))
        successi=rewardsumi>-100#reacher-150#push#shape 250
        tsrarray=np.vstack((tsrarray,successi))
        successratei=np.average(successi)#shape 1
        #print('successrate',successratei)
        rewardaveragei=np.average(rewardsumi)
        print('rewardaveragei',rewardaveragei)
        #print('shape',rewardsumi[-lastnum:].shape)#it is 50
        constri=np.load(os.path.join(logdir, "constr.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstri=np.sum(constri[0:fh])
        constrirate=totalconstri/constri.shape[0]
        cvarray=np.vstack((cvarray,constri))#reacher#np.zeros((1000,))#push#
        #print('constrirate',constrirate)
        constrcbfi=np.load(os.path.join(logdir, "constrsafety.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstrcbfi=np.sum(constrcbfi[0:fh])
        constrcbfirate=totalconstrcbfi/constrcbfi.shape[0]
        constrcbf2i=np.load(os.path.join(logdir, "constrsafety2.npy"))
        totalconstrcbf2i=np.sum(constrcbf2i[0:fh])
        constrcbf2irate=totalconstrcbf2i/constrcbf2i.shape[0]
        srlist.append(successratei)
        cvrlist.append(constrirate)
        cvrcbflist.append(constrcbfirate)
        cvrcbf2list.append(constrcbf2irate)
        #tsrlist.append(tasksuccirate)
        ralist.append(rewardaveragei)

    #calculate the statistics: mean and std
    rfarray=rfarray[1:]
    cvarray=cvarray[1:]
    tsrarray=tsrarray[1:]
    cvcarray=np.cumsum(cvarray,axis=1) 
    #print('rfarray.shape',rfarray.shape)#3,250
    rfmean=np.mean(rfarray,axis=0)
    cvcmean=np.mean(cvcarray,axis=0)
    tsrmean=np.mean(tsrarray,axis=0)
    #print(rfmean)
    rfstd=np.std(rfarray,axis=0)
    cvcstd=np.std(cvcarray,axis=0)
    tsrstd=np.std(tsrarray,axis=0)
    lenseed=len(seedlist)
    '''
    pu.simple_plot(rfmean, std=rfstd, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed, 'rewards'+str(lenseed)+'trajs'+date+'-'+time+'epochs'+str(fh)+'.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot(tsrmean, std=tsrstd, title='Average task success rate',
                            file=os.path.join(logdirbeforeseed, 'tsr'+str(lenseed)+'trajs'+date+'-'+time+'epochs'+str(fh)+'.pdf'),
                            ylabel='Average task success rate', xlabel='# Training updates')
    '''
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
    '''
    pu.simple_plot(rfcmean, std=rfcstd, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed, 'rewards'+str(lenseed)+'epochs'+date+'-'+time+'epochs'+str(fh)+'.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot(tsrcmean, std=tsrcstd, title='Average task success rate',
                            file=os.path.join(logdirbeforeseed, 'tsrc'+str(lenseed)+'epochs'+date+'-'+time+'epochs'+str(fh)+'.pdf'),
                            ylabel='Average  task success rate', xlabel='# Training updates')
    '''


    logdirbeforeseed2 = os.path.join(outputdir+date2,time2) #params['logdir']#around line 35
    #logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
    print('logdirbeforeseed2',logdirbeforeseed2)
    srlist2=[]#success rate list
    ralist2=[]#reward average list
    fh2=500#250#fh means 500
    rfarray2=np.zeros((fh2,))#reacher#np.zeros((1000,))#push#
    cvarray2=np.zeros((fh2,))#reacher#np.zeros((1000,))#push#
    cvrlist2=[]#constraint violation rate list
    cvrcbflist2=[]#constraint violation rate list
    cvrcbf2list2=[]#constraint violation rate list
    tsrarray2=np.zeros((fh2,))#reacher#np.zeros((1000,))#push#
    seedlist2=[1,2,3,4,5,6,7,8,9,10]#[1,2,3]#[1,2]#[1,2,3,4,5]#24,25#[1,101,201]#22#[4,5,6,7,8,9,10]#23#[1,26,51]##

    for seed in seedlist2:
        logdir2=os.path.join(logdirbeforeseed2, str(seed))
        #update_dir = os.path.join(logdir, "update_%d" % i)#
        rewardi2=np.load(os.path.join(logdir2, "rewards.npy"))
        #print('rewardi.shape',rewardi.shape)#250,100
        rewardsumi2=np.sum(rewardi2[0:fh2],axis=1)
        #print('rewardsumi.shape',rewardsumi.shape)#250
        rfarray2=np.vstack((rfarray2,rewardsumi2))
        successi2=rewardsumi2>-100#reacher-150#push#shape 250
        tsrarray2=np.vstack((tsrarray2,successi2))
        successratei2=np.average(successi2)#shape 1
        #print('successrate',successratei)
        rewardaveragei2=np.average(rewardsumi2)
        print('rewardaveragei',rewardaveragei2)
        #print('shape',rewardsumi[-lastnum:].shape)#it is 50
        constri2=np.load(os.path.join(logdir2, "constr.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstri2=np.sum(constri2[0:fh2])
        constrirate2=totalconstri2/constri2.shape[0]
        cvarray2=np.vstack((cvarray2,constri2))#reacher#np.zeros((1000,))#push#
        #print('constrirate',constrirate)
        constrcbfi2=np.load(os.path.join(logdir2, "constrcbf.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstrcbfi2=np.sum(constrcbfi2[0:fh2])
        constrcbfirate2=totalconstrcbfi2/constrcbfi2.shape[0]
        constrcbf2i2=np.load(os.path.join(logdir2, "constrcbf2.npy"))
        totalconstrcbf2i2=np.sum(constrcbf2i2[0:fh2])
        constrcbf2irate2=totalconstrcbf2i2/constrcbf2i2.shape[0]
        srlist2.append(successratei2)
        cvrlist2.append(constrirate2)
        cvrcbflist2.append(constrcbfirate2)
        cvrcbf2list2.append(constrcbf2irate2)
        #tsrlist.append(tasksuccirate)
        ralist2.append(rewardaveragei2)

    #calculate the statistics: mean and std
    rfarray2=rfarray2[1:]
    cvarray2=cvarray2[1:]
    tsrarray2=tsrarray2[1:]
    cvcarray2=np.cumsum(cvarray2,axis=1) 
    #print('rfarray.shape',rfarray.shape)#3,250
    rfmean2=np.mean(rfarray2,axis=0)
    cvcmean2=np.mean(cvcarray2,axis=0)
    tsrmean2=np.mean(tsrarray2,axis=0)
    #print(rfmean)
    rfstd2=np.std(rfarray2,axis=0)
    cvcstd2=np.std(cvcarray2,axis=0)
    tsrstd2=np.std(tsrarray2,axis=0)
    #print(rfstd)
    lenseed2=len(seedlist2)
    pu.simple_plot2(rfmean, std=rfstd, data2=rfmean2, std2=rfstd2,title='Average Rewards',
                            file=os.path.join(logdirbeforeseed2, 'rewards'+str(lenseed2)+'trajs'+date2+'-'+time2+'epochs'+str(fh2)+'compare.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot2(cvcmean, std=cvcstd,data2=cvcmean2, std2=cvcstd2, title='Constraint Violations',
                            file=os.path.join(logdirbeforeseed2, 'cvc'+str(lenseed2)+'trajs'+date2+'-'+time2+'epochs'+str(fh2)+'compare.pdf'),
                            ylabel='Cumulative violations', xlabel='# Trajectories')
    #pu.simple_plot(tsrmean2, std=tsrstd2, title='Average task success rate',
                            #file=os.path.join(logdirbeforeseed2, 'tsr'+str(lenseed2)+'trajs'+date2+'-'+time2+'epochs'+str(fh2)+'compare.pdf'),
                            #ylabel='Average task success rate', xlabel='# Training updates')
    rfcarray2=np.zeros((lenseed2,))#c means corse
    tsrcarray2=np.zeros((lenseed2,))#c means corse
    for i in range(int(rfarray2.shape[1]/10)):
        rfciclip2=rfarray2[:,i*10:(i+1)*10]
        tsrciclip2=tsrarray2[:,i*10:(i+1)*10]
        #print(rfciclip)
        rfci2=np.average(rfciclip2,axis=1)
        tsrci2=np.average(tsrciclip2,axis=1)
        #print('rfci.shape',rfci)
        rfcarray2=np.vstack((rfcarray2,rfci2))
        tsrcarray2=np.vstack((tsrcarray2,tsrci2))
    rfcarray2=rfcarray2[1:]
    tsrcarray2=tsrcarray2[1:]
    #print('rfcarray.shape',rfcarray.shape)#3,250
    rfcmean2=np.mean(rfcarray2,axis=1)
    tsrcmean2=np.mean(tsrcarray2,axis=1)
    #print(rfcmean)
    rfcstd2=np.std(rfcarray2,axis=1)
    tsrcstd2=np.std(tsrcarray2,axis=1)
    #print(rfcstd)
    pu.simple_plot2(rfcmean, std=rfcstd,data2=rfcmean2, std2=rfcstd2, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed2, 'rewards'+str(lenseed2)+'epochs'+date2+'-'+time2+'epochs'+str(fh2)+'compare.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    #pu.simple_plot(tsrcmean, std=tsrcstd, title='Average task success rate',
                            #file=os.path.join(logdirbeforeseed, 'tsrc'+str(lenseed)+'epochs'+date2+'-'+time2+'epochs'+str(fh)+'.pdf'),
                            #ylabel='Average  task success rate', xlabel='# Training updates')
    

    '''
    sra=np.array(srlist)
    cvra=np.array(cvrlist)
    cvrcbfa=np.array(cvrcbflist)
    cvrcbf2a=np.array(cvrcbf2list)
    #tsra=np.array(tsrlist)
    raa=np.array(ralist)
    ralasta=np.array(ralastlist)

    sraave=np.mean(sra)
    srastd=np.std(sra)
    print('successrate ave',sraave,'successrate std',srastd)
    pu.simple_plot(sra, title='Success rate %f'%(sraave)+"\u00B1"+'%f'%(srastd),
                            file=os.path.join(logdirbeforeseed, 'success'+str(lenseed)+'rate'+date+'-'+time+'epochs'+str(fh)+'.pdf'),
                            ylabel='success rate', xlabel='# seeds',nonreward=True)
    cvraave=np.mean(cvra)
    cvrastd=np.std(cvra)
    print('constraint rate ave',cvraave,'constraint rate std',cvrastd)
    pu.simple_plot(cvra, title='Constraint violation rate %f'%(cvraave)+"\u00B1"+'%f'%(cvrastd),
                            file=os.path.join(logdirbeforeseed, 'violation'+str(lenseed)+'rate'+date+'-'+time+'epochs'+str(fh)+'.pdf'),
                            ylabel='constraint violation rate', xlabel='# seeds',nonreward=True)
    cvrcbfaave=np.mean(cvrcbfa)
    cvrcbfastd=np.std(cvrcbfa)
    print('constraint rate cbf ave',cvrcbfaave,'constraint rate cbf std',cvrcbfastd)
    pu.simple_plot(cvrcbfa, title='Constraint violation cbf rate %f'%(cvrcbfaave)+"\u00B1"+'%f'%(cvrcbfastd),
                            file=os.path.join(logdirbeforeseed, 'violation'+str(lenseed)+'rate'+date+'-'+time+'epochs'+str(fh)+'cbf.pdf'),
                            ylabel='constraint violation cbf rate', xlabel='# seeds',nonreward=True)
    cvrcbf2aave=np.mean(cvrcbf2a)
    cvrcbf2astd=np.std(cvrcbf2a)
    print('constraint rate cbf2 ave',cvrcbf2aave,'constraint rate cbf2 std',cvrcbfastd)
    pu.simple_plot(cvrcbf2a, title='Constraint violation cbf2 rate %f'%(cvrcbf2aave)+"\u00B1"+'%f'%(cvrcbf2astd),
                            file=os.path.join(logdirbeforeseed, 'violation'+str(lenseed)+'rate'+date+'-'+time+'epochs'+str(fh)+'cbf2.pdf'),
                            ylabel='constraint violation cbf2 rate', xlabel='# seeds',nonreward=True)
    print('reward ave',np.mean(raa),'reward std',np.std(raa))
    print('reward last ave',np.mean(ralasta),'reward last std',np.std(ralasta))
    #making plots
    '''

if __name__=='__main__':
    main()