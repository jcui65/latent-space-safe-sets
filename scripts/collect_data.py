import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/cuijin/Project6remote/latent-space-safe-sets')

#from latentsafesets.utils.arg_parser import parse_args
from latentsafesets.utils.arg_parser_reacher import parse_args
import latentsafesets.utils as utils
from latentsafesets.utils.teacher import ConstraintTeacher, ReacherTeacher,\
    ReacherConstraintTeacher, StrangeTeacher, PushTeacher, OutburstPushTeacher, \
    SimplePointBotTeacher, ReacherConstraintdense1Teacher, ReacherConstraintdense2Teacher
import latentsafesets.utils.plot_utils as pu
import sympy
import logging
import os
import numpy as np
log = logging.getLogger("collect")


env_teachers = {#it is a dictionary, right?
    'spb': [
        SimplePointBotTeacher, ConstraintTeacher, StrangeTeacher
    ],
    'reacher': [
        ReacherTeacher,ReacherConstraintdense1Teacher,ReacherConstraintdense2Teacher,StrangeTeacher#ReacherConstraintTeacher,
    ],
    'push': [
        PushTeacher, OutburstPushTeacher
    ],
}


def generate_teacher_demo_data(env, data_dir, teacher, n=100, noisy=False, logdir=None):
    log.info("Generating teacher demo trajectories")
    file = os.path.join('data', data_dir)#create the data folder!
    if not os.path.exists(file):#['SimplePointBot','SimplePointBotConstraints',]#
        os.makedirs(file)#data/SimplePointBot or data/SimplePointBotConstraints
    else:
        raise RuntimeError("Directory %s already exists." % file)#not good code writing
    teacher = teacher(env, noisy=noisy)#SimplePointBotTeacher, or ConstraintTeacher,
    demonstrations = []#an empty list
    for i in range(n):
        traj = teacher.generate_trajectory()#around line 33 in teacher.py
        reward = sum([frame['reward'] for frame in traj])#traj is a list of dictionaries
        #why not directly use rtg[0]?
        print('Trajectory %d, Reward %d' % (i, reward))
        demonstrations.append(traj)#traj is one piece of trajectories
        utils.save_trajectory(traj, file, i)#around line 86 in utils.py#save 1 piece of traj
        if i < 50 and logdir is not None:
            pu.make_movie(traj, os.path.join(logdir, '%s_%d.gif' % (data_dir, i)))
    return demonstrations#list of list of trajectories?

def generate_teacher_demo_datasafety(env, data_dir, teacher, n=100, noisy=False, logdir=None):
    log.info("Generating teacher demo trajectories")
    file = os.path.join('data', data_dir)#create the data folder!
    if not os.path.exists(file):#['SimplePointBot','SimplePointBotConstraints',]#
        os.makedirs(file)#data/SimplePointBot or data/SimplePointBotConstraints
    else:
        raise RuntimeError("Directory %s already exists." % file)#not good code writing
    teacher = teacher(env, noisy=noisy)#SimplePointBotTeacher, or ConstraintTeacher,
    demonstrations = []#an empty list
    #ro=0.04#0.05-1
    for i in range(n):#generate n trajectories
        print('data_dir',data_dir)
        if data_dir=='ReacherConstraintdense1' or data_dir=='ReacherConstraintdense2':
            #print('teacherenter',teacher)
            angled=np.pi/2 - 1.5*np.pi*i/n#d means desired
            if data_dir=='ReacherConstraintdense1':
                radius=0.045#0.05#0.04#
            else:
                radius=0.12
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
            #print('s1t2',s1t2)#print('s1t1',s1[t1])
            traj = teacher.generate_trajectorysafety_dense(xa=s0t1,ya=s0t2,xa2=s1t1,ya2=s1t2,angled=angled)
        else:
            traj = teacher.generate_trajectorysafety()#line 33 in teacher.py#100 transitions
        reward = sum([frame['reward'] for frame in traj])#traj is a list of dictionaries
        #why not directly use rtg[0]?
        print('Trajectory %d, Reward %d' % (i, reward))
        demonstrations.append(traj)#traj is one piece of trajectories
        utils.save_trajectory(traj, file, i)#86 in utils.py#save 1 traj having 100 steps
        '''
        for k,frame in enumerate(traj):
            st=frame['state']
            ns=frame['next_state']
            #print('state',state,'next_state',ns)
            log.info('%dstate%d: 0:%f,1:%f,2:%f,3:%f,4:%f,5:%f,' % (i,k,st[0],st[1],st[2],st[3],st[4],st[5]))
            log.info('06:%f,07:%f,08:%f,09:%f,10:%f,11:%f,12:%f,' % (st[6],st[7],st[8],st[9],st[10],st[11],st[12]))
            log.info('13:%f,14:%f,15:%f,16:%f,17:%f,18:%f,19:%f,' % (st[13],st[14],st[15],st[16],st[17],st[18],st[19]))
            log.info('20:%f,21:%f,22:%f,23:%f,24:%f,25:%f,26:%f.' % (st[20],st[21],st[22],st[23],st[24],st[25],st[26]))
            log.info('%dnext_state%d: 0:%f,1:%f,2:%f,3:%f,4:%f,5:%f,' % (i,k,ns[0],ns[1],ns[2],ns[3],ns[4],ns[5]))
            log.info('06:%f,07:%f,08:%f,09:%f,10:%f,11:%f,12:%f,' % (ns[6],ns[7],ns[8],ns[9],ns[10],ns[11],ns[12]))
            log.info('13:%f,14:%f,15:%f,16:%f,17:%f,18:%f,19:%f,' % (ns[13],ns[14],ns[15],ns[16],ns[17],ns[18],ns[19]))
            log.info('20:%f,21:%f,22:%f,23:%f,24:%f,25:%f,26:%f.' % (ns[20],ns[21],ns[22],ns[23],ns[24],ns[25],ns[26]))
        '''
        if i < 50 and logdir is not None:
            pu.make_movie(traj, os.path.join(logdir, '%s_%d.gif' % (data_dir, i)))#do I add relative here or in other place?
    return demonstrations#list of list of trajectories?

def generate_teacher_demo_datasafety_relative(env, data_dir, teacher, n=100, noisy=False, logdir=None):
    log.info("Generating teacher demo trajectories")
    file = os.path.join('data_relative', data_dir)#create the data folder!
    if not os.path.exists(file):#['SimplePointBot','SimplePointBotConstraints',]#
        os.makedirs(file)#data/SimplePointBot or data/SimplePointBotConstraints
    else:
        raise RuntimeError("Directory %s already exists." % file)#not good code writing
    teacher = teacher(env, noisy=noisy)#SimplePointBotTeacher, or ConstraintTeacher,
    demonstrations = []#an empty list
    for i in range(n):
        traj = teacher.generate_trajectorysafety_relative()#line 33 in teacher.py#100 transitions
        reward = sum([frame['reward'] for frame in traj])#traj is a list of dictionaries
        #why not directly use rtg[0]?
        print('Trajectory %d, Reward %d' % (i, reward))
        demonstrations.append(traj)#traj is one piece of trajectories
        utils.save_trajectory_relative(traj, file, i)#86 in utils.py#save 1 traj having 100 steps
        if i < 50 and logdir is not None:
            pu.make_movie_relative(traj, os.path.join(logdir, '%s_%d.gif' % (data_dir, i)))#do I add relative here or in other place?
    return demonstrations#list of list of trajectories?

def main():
    params = parse_args()

    logdir = utils.get_file_prefix()#around line 46 in utils.py
    os.makedirs(logdir)
    utils.init_logging(logdir)#around line 58 in utils.py

    env = utils.make_env(params)#around line 153#SimplePointBot
    print('horizon',env.horizon)#horizon 100, that is what will be printed!

    teachers = env_teachers[params['env']]#[SimplePointBotTeacher, ConstraintTeacher, StrangeTeacher]
    data_dirs = params['data_dirs']#['SimplePointBot','SimplePointBotConstraints',]
    data_counts = params['data_counts']#[50,50] for spb

    for teacher, data_dir, count in list(zip(teachers, data_dirs, data_counts)):
        #still 2, always take the least number: https://www.programiz.com/python-programming/methods/built-in/zip
        #generate_teacher_demo_data(env, data_dir, teacher, count, True, logdir)#see around 31#it's with noise in action
        generate_teacher_demo_datasafety(env, data_dir, teacher, count, True, logdir)  # see around 31
        #generate_teacher_demo_datasafety_relative(env, data_dir, teacher, count, True, logdir)  # see around 31


if __name__ == '__main__':
    main()
