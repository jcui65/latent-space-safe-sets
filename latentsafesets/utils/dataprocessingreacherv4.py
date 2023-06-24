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
@click.option('--date1', default='05-16',help='the date when the simulation started', type=str)#'05-11'
@click.option('--time1', default='23-10-18', help='time of the simulation', type=str)#'02-13-52'
@click.option('--date2', default='05-17',help='the date when the simulation started', type=str)#'05-11'
@click.option('--time2', default='22-01-54', help='time of the simulation', type=str)#'18-15-09'
@click.option('--date3', default='05-17',help='the date when the simulation started', type=str)#'05-11'
@click.option('--time3', default='12-19-57', help='time of the simulation', type=str)#'22-33-19'
@click.option('--date4', default='05-17',help='the date when the simulation started', type=str)#'05-11'
@click.option('--time4', default='16-23-39', help='time of the simulation', type=str)#'22-33-23'
@click.option('--fh', default=500, help='five hundred or 250', type=int)


def main(date1, time1,date2, time2,date3,time3,date4,time4,fh):

    def data_loading(logdirbeforeseed,lenseed):
        rfarray=np.load(os.path.join(logdirbeforeseed, 'rewards'+str(lenseed)+'.npy'))
        cvarray=np.load(os.path.join(logdirbeforeseed, 'cv'+str(lenseed)+'.npy'))
        tsrarray=np.load(os.path.join(logdirbeforeseed, 'tsr'+str(lenseed)+'.npy'))
        cvcarray=np.load(os.path.join(logdirbeforeseed, 'cvc'+str(lenseed)+'.npy'))
        rfmean=np.load(os.path.join(logdirbeforeseed, 'rewardsmean'+str(lenseed)+'.npy'))
        cvcmean=np.load(os.path.join(logdirbeforeseed, 'cvcmean'+str(lenseed)+'.npy'))
        tsrmean=np.load(os.path.join(logdirbeforeseed, 'tsrmean'+str(lenseed)+'.npy'))
        rfstd=np.load(os.path.join(logdirbeforeseed, 'rewardsstd'+str(lenseed)+'.npy'))
        cvcstd=np.load(os.path.join(logdirbeforeseed, 'cvcstd'+str(lenseed)+'.npy'))
        tsrstd=np.load(os.path.join(logdirbeforeseed, 'tsrstd'+str(lenseed)+'.npy'))
        rfcarray=np.load(os.path.join(logdirbeforeseed, 'rewardsc'+str(lenseed)+'.npy'))
        tsrcarray=np.load(os.path.join(logdirbeforeseed, 'tsrc'+str(lenseed)+'.npy'), )
        rfcmean=np.load(os.path.join(logdirbeforeseed, 'rewardscmean'+str(lenseed)+'.npy'))
        tsrcmean=np.load(os.path.join(logdirbeforeseed, 'tsrcmean'+str(lenseed)+'.npy'))
        rfcstd=np.load(os.path.join(logdirbeforeseed, 'rewardscstd'+str(lenseed)+'.npy'))
        tsrcstd=np.load(os.path.join(logdirbeforeseed, 'tsrcstd'+str(lenseed)+'.npy'))
        #return rfarray,cvarray,tsrarray,cvcarray,rfmean,cvcmean,tsrmean,rfstd,cvcstd,tsrstd
        return rfmean,cvcmean,tsrmean,rfcmean,tsrcmean,rfstd,cvcstd,tsrstd,rfcstd,tsrcstd


    outputdir='/home/cuijin/Project6remote/latent-space-safe-sets/outputs/2023-'
    logdirbeforeseed1 = os.path.join(outputdir+date1,time1) #params['logdir']#around line 35
    print('logdirbeforeseed1',logdirbeforeseed1)
    logdirbeforeseed2 = os.path.join(outputdir+date2,time2) #params['logdir']#around line 35
    print('logdirbeforeseed2',logdirbeforeseed2)
    logdirbeforeseed3 = os.path.join(outputdir+date3,time3) #params['logdir']#around line 35
    print('logdirbeforeseed3',logdirbeforeseed3)
    logdirbeforeseed4 = os.path.join(outputdir+date4,time4) #params['logdir']#around line 35
    print('logdirbeforeseed4',logdirbeforeseed4)
    seedlist=[1,2,3]#[1,2]#[1,2,3,4,5]#24,25#[1,2,3,4,5,6,7,8,9,10]#[1,101,201]#22#[4,5,6,7,8,9,10]#23#[1,26,51]##

    lenseed=10#3#len(seedlist)
    lenseed2=3#10#
    rfmean1,cvcmean1,tsrmean1,rfcmean1,tsrcmean1,rfstd1,cvcstd1,tsrstd1,rfcstd1,tsrcstd1=data_loading(logdirbeforeseed1,lenseed2)
    rfmean2,cvcmean2,tsrmean2,rfcmean2,tsrcmean2,rfstd2,cvcstd2,tsrstd2,rfcstd2,tsrcstd2=data_loading(logdirbeforeseed2,lenseed)
    rfmean3,cvcmean3,tsrmean3,rfcmean3,tsrcmean3,rfstd3,cvcstd3,tsrstd3,rfcstd3,tsrcstd3=data_loading(logdirbeforeseed3,lenseed2)
    #rfmean4,cvcmean4,tsrmean4,rfcmean4,tsrcmean4,rfstd4,cvcstd4,tsrstd4,rfcstd4,tsrcstd4=data_loading(logdirbeforeseed4,lenseed)
    
    rfmean4,cvcmean4,tsrmean4,rfcmean4,tsrcmean4,rfstd4,cvcstd4,tsrstd4,rfcstd4,tsrcstd4=None,None,None,None,None,None,None,None,None,None


    pu.simple_plot4(rfmean1,rfmean2, rfmean3,rfmean4, std=rfstd1, std2=rfstd2,std3=rfstd3, std4=rfstd4, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed3, 'rewards'+str(lenseed)+'trajs'+date3+'-'+time3+'epochs'+str(fh)+'compare.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot4(cvcmean1, cvcmean2, cvcmean3, cvcmean4, std=cvcstd1,std2=cvcstd2,std3=cvcstd3,std4=cvcstd4, title='Constraint Violations',
                            file=os.path.join(logdirbeforeseed3, 'cvc'+str(lenseed)+'trajs'+date3+'-'+time3+'epochs'+str(fh)+'compare.pdf'),
                            ylabel='No. of constraint violations', xlabel='# Trajectories')
    pu.simple_plot4(rfcmean1,rfcmean2, rfcmean3,rfcmean4, std=rfcstd1, std2=rfcstd2,std3=rfcstd3, std4=rfcstd4, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed3, 'rewards'+str(lenseed)+'epochs'+date3+'-'+time3+'epochs'+str(fh)+'compare.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    #pu.simple_plot(tsrmean, std=tsrstd, title='Average task success rate',
                            #file=os.path.join(logdirbeforeseed4, 'tsr'+str(lenseed)+'trajs'+date+'-'+time+'epochs'+str(fh)+'.pdf'),
                            #ylabel='Average task success rate', xlabel='# Training updates')
    
    
    '''
    rfcarray=np.zeros((lenseed,))#c means corse
    tsrcarray=np.zeros((lenseed,))#c means corse

    rfcarray=rfcarray[1:]
    tsrcarray=tsrcarray[1:]

    #print('rfcarray.shape',rfcarray.shape)#3,250
    rfcmean=np.mean(rfcarray,axis=1)
    tsrcmean=np.mean(tsrcarray,axis=1)
    
    #print(rfcmean)
    rfcstd=np.std(rfcarray,axis=1)
    tsrcstd=np.std(tsrcarray,axis=1)

    #print(rfcstd)#
    pu.simple_plot(rfcmean, std=rfcstd, title='Average Rewards',
                            file=os.path.join(logdirbeforeseed, 'rewards'+str(lenseed)+'epochs'+date+'-'+time+'epochs'+str(fh)+'.pdf'),
                            ylabel='Average Reward', xlabel='# Training updates')
    pu.simple_plot(tsrcmean, std=tsrcstd, title='Average task success rate',
                            file=os.path.join(logdirbeforeseed, 'tsrc'+str(lenseed)+'epochs'+date+'-'+time+'epochs'+str(fh)+'.pdf'),
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