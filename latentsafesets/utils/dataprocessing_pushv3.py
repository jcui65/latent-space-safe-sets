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
@click.option('--date1', default='04-04',help='the date when the simulation started', type=str)
@click.option('--time1', default='13-27-02', help='time of the simulation', type=str)
@click.option('--date2', default='05-08',help='the date when the simulation started', type=str)
@click.option('--time2', default='01-09-06', help='time of the simulation', type=str)
def main(date1, time1,date2, time2):
    outputdir='/home/cuijin/Project6remote/latent-space-safe-sets/outputs/2023-'
    #date=mar28#mar26#mar27#mar25#mar23#mar22#mar24#
    #time='20-29-18'#'20-28-35'#'20-26-13'#'20-23-19'#'00-08-25'#'00-04-54'#'00-02-15'#'20-22-45'#'20-22-06'#'20-11-00'#'20-09-26'#'20-07-08'#'20-05-38'#'19-38-18'#'15-36-29'#'15-35-54'#'15-22-41'#'14-54-22'#'14-53-10'#'18-37-20'#'17-27-44'#'17-06-29'#
    #time='15-53-12'#'09-30-18'##'11-53-02'#'11-51-10'#'11-34-01'#'11-33-24'#'13-24-32'#'19-45-12'#'14-02-35'#'14-03-38'#'13-24-46'#'00-54-16'#'19-45-47'#
    logdirbeforeseed = os.path.join(outputdir+date1,time1) #params['logdir']#around line 35
    #logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
    print('logdirbeforeseed',logdirbeforeseed)
    srlist=[]#success rate list
    ralist=[]#reward average list
    fh1=1000#500#fh means five hundred
    rfarray=np.zeros((fh1,))#push#np.zeros((250,))#reacher
    cvarray=np.zeros((fh1,))#reacher#np.zeros((1000,))#push#
    tsrarray=np.zeros((fh1,))#reacher#np.zeros((1000,))#push#
    ralastlist=[]#reward average last list
    cvrlist=[]#constraint violation rate list
    cvrcbflist=[]#constraint violation rate list
    cvrcbf2list=[]#constraint violation rate list
    lastnum=50
    seedlist=[1,2,3,4,5,6,7,8,9,10]#[1,2,3]#24,25#[4,5,6,7,8,9,10]#[1,101,201]#23#22#[1,26,51]##
    for seed in seedlist:
        logdir=os.path.join(logdirbeforeseed, str(seed))
        #update_dir = os.path.join(logdir, "update_%d" % i)#
        rewardi=np.load(os.path.join(logdir, "rewards.npy"))
        #print('rewardi.shape',rewardi.shape)#250,100
        rewardsumi=np.sum(rewardi,axis=1)
        #print('rewardsumi.shape',rewardsumi.shape)#250
        rfarray=np.vstack((rfarray,rewardsumi))
        successi=rewardsumi>-150#push#-100#reacher
        tsrarray=np.vstack((tsrarray,successi))
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
        cvarray=np.vstack((cvarray,constri))#reacher#np.zeros((1000,))#push#
        constrcbfi=np.load(os.path.join(logdir, "constrcbf.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstrcbfi=np.sum(constrcbfi)
        constrcbfirate=totalconstrcbfi/constrcbfi.shape[0]
        constrcbf2i=np.load(os.path.join(logdir, "constrcbf2.npy"))
        totalconstrcbf2i=np.sum(constrcbf2i)
        constrcbf2irate=totalconstrcbf2i/constrcbf2i.shape[0]
        #print('constrirate',constrirate)
        srlist.append(successratei)
        cvrlist.append(constrirate)
        cvrcbflist.append(constrcbfirate)
        cvrcbf2list.append(constrcbf2irate)
        ralist.append(rewardaveragei)
        ralastlist.append(rewardaveragelasti)

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


    logdirbeforeseed2 = os.path.join(outputdir+date2,time2) #params['logdir']#around line 35
    print('logdirbeforeseed2',logdirbeforeseed2)
    srlist2=[]#success rate list
    ralist2=[]#reward average list
    fh2=1000#500#fh means five hundred
    rfarray2=np.zeros((fh2,))#push#np.zeros((250,))#reacher
    cvarray2=np.zeros((fh2,))#reacher#np.zeros((1000,))#push#
    tsrarray2=np.zeros((fh2,))#reacher#np.zeros((1000,))#push#
    ralastlist2=[]#reward average last list
    cvrlist2=[]#constraint violation rate list
    cvrcbflist2=[]#constraint violation rate list
    cvrcbf2list2=[]#constraint violation rate list
    seedlist2=[1,2,3]#24,25#[4,5,6,7,8,9,10]#[1,101,201]#23#22#[1,26,51]##
    for seed in seedlist2:
        logdir2=os.path.join(logdirbeforeseed2, str(seed))
        #update_dir = os.path.join(logdir, "update_%d" % i)#
        rewardi2=np.load(os.path.join(logdir2, "rewards.npy"))
        #print('rewardi.shape',rewardi.shape)#250,100
        rewardsumi2=np.sum(rewardi2,axis=1)
        #print('rewardsumi.shape',rewardsumi.shape)#250
        rfarray2=np.vstack((rfarray2,rewardsumi2))
        successi2=rewardsumi2>-150#push#-100#reacher
        tsrarray2=np.vstack((tsrarray2,successi2))
        successratei2=np.average(successi2)
        #print('successrate',successratei)
        rewardaveragei2=np.average(rewardsumi2)
        print('rewardaveragei2',rewardaveragei2)
        #print('shape',rewardsumi[-lastnum:].shape)#it is 50
        constri2=np.load(os.path.join(logdir2, "constr.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstri2=np.sum(constri2)
        constrirate2=totalconstri2/constri2.shape[0]
        cvarray2=np.vstack((cvarray2,constri2))#reacher#np.zeros((1000,))#push#
        constrcbfi2=np.load(os.path.join(logdir, "constrcbf.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstrcbfi2=np.sum(constrcbfi2)
        constrcbfirate2=totalconstrcbfi2/constrcbfi2.shape[0]
        constrcbf2i2=np.load(os.path.join(logdir, "constrcbf2.npy"))
        totalconstrcbf2i2=np.sum(constrcbf2i2)
        constrcbf2irate2=totalconstrcbf2i2/constrcbf2i2.shape[0]
        #print('constrirate',constrirate)
        srlist2.append(successratei2)
        cvrlist2.append(constrirate2)
        cvrcbflist2.append(constrcbfirate2)
        cvrcbf2list2.append(constrcbf2irate2)
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






    pu.simple_plot2(rfmean, std=rfstd,data2=rfmean2, std2=rfstd2, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed2, 'rewards'+str(lenseed2)+'trajs'+date2+'-'+time2+'compare.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot2(cvcmean, std=cvcstd,data2=cvcmean2, std2=cvcstd2, title='Constraint Violations',
                            file=os.path.join(logdirbeforeseed2, 'cvc'+str(lenseed2)+'trajs'+date2+'-'+time2+'epochs'+str(fh2)+'compare.pdf'),
                            ylabel='Cumulative violations', xlabel='# Trajectories')
    #pu.simple_plot(tsrmean, std=tsrstd, title='Average task success rate',
                            #file=os.path.join(logdirbeforeseed, 'tsr'+str(lenseed)+'trajs'+date2+'-'+time2+'.pdf'),
                            #ylabel='Average task success rate', xlabel='# Training updates')
    
    rfcarray2=np.zeros((lenseed2,))#c means corse
    tsrcarray2=np.zeros((lenseed2,))#c means corse
    for i in range(int(rfarray.shape[1]/10)):
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
                            file=os.path.join(logdirbeforeseed2, 'rewards'+str(lenseed2)+'epochs'+date2+'-'+time2+'compare.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    
    '''
    pu.simple_plot(tsrcmean, std=tsrcstd, title='Average task success rate',
                            file=os.path.join(logdirbeforeseed, 'tsrc'+str(lenseed)+'epochs'+date+'-'+time+'.pdf'),
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
                            file=os.path.join(logdirbeforeseed, 'success'+str(lenseed)+'rate'+date+'-'+time+'.pdf'),
                            ylabel='success rate', xlabel='# seeds',nonreward=True)
    cvraave=np.mean(cvra)
    cvrastd=np.std(cvra)
    print('constraint rate ave',cvraave,'constraint rate std',cvrastd)
    pu.simple_plot(cvra, title='Constraint violation rate %f'%(cvraave)+"\u00B1"+'%f'%(cvrastd),
                            file=os.path.join(logdirbeforeseed, 'violation'+str(lenseed)+'rate'+date+'-'+time+'.pdf'),
                            ylabel='constraint violation rate', xlabel='# seeds',nonreward=True)
    cvrcbfaave=np.mean(cvrcbfa)
    cvrcbfastd=np.std(cvrcbfa)
    print('constraint rate cbf ave',cvrcbfaave,'constraint rate cbf std',cvrcbfastd)
    pu.simple_plot(cvrcbfa, title='Constraint violation cbf rate %f'%(cvrcbfaave)+"\u00B1"+'%f'%(cvrcbfastd),
                            file=os.path.join(logdirbeforeseed, 'violation'+str(lenseed)+'rate'+date+'-'+time+'cbf.pdf'),
                            ylabel='constraint violation cbf rate', xlabel='# seeds',nonreward=True)
    cvrcbf2aave=np.mean(cvrcbf2a)
    cvrcbf2astd=np.std(cvrcbf2a)
    print('constraint rate cbf2 ave',cvrcbf2aave,'constraint rate cbf2 std',cvrcbfastd)
    pu.simple_plot(cvrcbf2a, title='Constraint violation cbf2 rate %f'%(cvrcbf2aave)+"\u00B1"+'%f'%(cvrcbf2astd),
                            file=os.path.join(logdirbeforeseed, 'violation'+str(lenseed)+'rate'+date+'-'+time+'cbf2.pdf'),
                            ylabel='constraint violation cbf2 rate', xlabel='# seeds',nonreward=True)
    print('reward ave',np.mean(raa),'reward std',np.std(raa))
    print('reward last ave',np.mean(ralasta),'reward last std',np.std(ralasta))
    #making plots
    '''
if __name__=='__main__':
    main()