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
@click.option('--date', default='04-14',help='the date when the simulation started', type=str)
@click.option('--time', default='02-41-37', help='time of the simulation', type=str)

def main(date, time):
    outputdir='/home/cuijin/Project6remote/latent-space-safe-sets/outputs/2023-'
    mar24='03-24'
    mar23='03-23'
    mar22='03-22'
    mar25='03-25'
    mar26='03-26'
    mar27='03-27'
    mar28='03-28'
    #date=mar28#mar22#mar24#mar26#mar27#mar25#mar23#
    #time='23-17-32'#'23-16-17'#'11-51-10'#'19-45-47'#'19-45-12'#'13-24-32'#'14-02-35'#'01-09-11'#'01-07-55'#'01-06-51'#'01-03-46'#'20-29-18'#'20-28-35'#'20-26-13'#'20-23-19'#'00-08-25'#'00-04-54'#'00-02-15'#
    #'20-22-45'#'20-22-06'#'20-11-00'#'20-09-26'#'20-07-08'#'20-05-38'#'19-38-18'#'15-36-29'#'15-35-54'#'15-22-41'#'14-54-22'#'14-53-10'#'18-37-20'#'17-27-44'#'17-06-29'#
    #time='15-53-12'#'09-30-18'##'11-53-02'#'11-34-01'#'11-33-24'#'14-03-38'#'13-24-46'#'00-54-16'#
    logdirbeforeseed = os.path.join(outputdir+date,time) #params['logdir']#around line 35
    #logdirbeforeseed = os.path.join('outputs/'+date,time) #params['logdir']#around line 35
    print('logdirbeforeseed',logdirbeforeseed)
    srlist=[]#success rate list
    ralist=[]#reward average list
    fh=500#250#fh means 500
    rfarray=np.zeros((fh,))#reacher#np.zeros((1000,))#push#
    ralastlist=[]#reward average last list
    cvrlist=[]#constraint violation rate list
    cvrcbflist=[]#constraint violation rate list
    cvrcbf2list=[]#constraint violation rate list
    tsrarray=np.zeros((fh,))#reacher#np.zeros((1000,))#push#
    #tsrlist=[]#constraint violation rate list
    lastnum=50
    seedlist=[1,2,3]#[1,2,3,4,5]#[1,2]#24,25#[1,2,3,4,5,6,7,8,9,10]#[1,101,201]#22#[4,5,6,7,8,9,10]#23#[1,26,51]##

    for seed in seedlist:
        logdir=os.path.join(logdirbeforeseed, str(seed))
        #update_dir = os.path.join(logdir, "update_%d" % i)#
        rewardi=np.load(os.path.join(logdir, "rewards.npy"))
        #print('rewardi.shape',rewardi.shape)#250,100
        rewardsumi=np.sum(rewardi,axis=1)
        #print('rewardsumi.shape',rewardsumi.shape)#250
        rfarray=np.vstack((rfarray,rewardsumi))
        successi=rewardsumi>-100#reacher-150#push#shape 250
        tsrarray=np.vstack((tsrarray,successi))
        successratei=np.average(successi)#shape 1
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
        constrcbfi=np.load(os.path.join(logdir, "constrcbf.npy"))
        #print('constri.shape',constri.shape)#250
        totalconstrcbfi=np.sum(constrcbfi)
        constrcbfirate=totalconstrcbfi/constrcbfi.shape[0]
        constrcbf2i=np.load(os.path.join(logdir, "constrcbf2.npy"))
        totalconstrcbf2i=np.sum(constrcbf2i)
        constrcbf2irate=totalconstrcbf2i/constrcbf2i.shape[0]
        #tasksucci=np.load(os.path.join(logdir, "tasksuccess.npy"))
        #print('constri.shape',constri.shape)#250
        #totaltasksucci=np.sum(tasksucci)
        #tasksuccirate=totaltasksucci/tasksucci.shape[0]
        #print('constrirate',constrirate)
        srlist.append(successratei)
        cvrlist.append(constrirate)
        cvrcbflist.append(constrcbfirate)
        cvrcbf2list.append(constrcbf2irate)
        #tsrlist.append(tasksuccirate)
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
    lenseed=len(seedlist)
    pu.simple_plot(rfmean, std=rfstd, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed, 'rewards'+str(lenseed)+'trajs'+date+'-'+time+'.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot(tsrmean, std=tsrstd, title='Average task success rate',
                            file=os.path.join(logdirbeforeseed, 'tsr'+str(lenseed)+'trajs'+date+'-'+time+'.pdf'),
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
                            file=os.path.join(logdirbeforeseed, 'rewards'+str(lenseed)+'epochs'+date+'-'+time+'.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot(tsrcmean, std=tsrcstd, title='Average task success rate',
                            file=os.path.join(logdirbeforeseed, 'tsrc'+str(lenseed)+'epochs'+date+'-'+time+'.pdf'),
                            ylabel='Average  task success rate', xlabel='# Training updates')
    sra=np.array(srlist)
    cvra=np.array(cvrlist)
    cvrcbfa=np.array(cvrcbflist)
    cvrcbf2a=np.array(cvrcbf2list)
    #tsra=np.array(tsrlist)
    raa=np.array(ralist)
    ralasta=np.array(ralastlist)
    #tsraave=np.mean(tsra)
    #tsrastd=np.std(tsra)
    #pu.simple_plot(tsraave, std=tsrastd, title='Average success ate',
                            #file=os.path.join(logdirbeforeseed, 'tsr'+str(lenseed)+'trajs'+date+'-'+time+'.pdf'),
                            #ylabel='Average success rate', xlabel='# Training updates')
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

if __name__=='__main__':
    main()