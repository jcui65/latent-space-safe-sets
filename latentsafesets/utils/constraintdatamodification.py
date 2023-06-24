import latentsafesets.utils as utils
import os
num_traj=50#500#300#100#
horizon=100#150#for pushing
directory='SimplePointBot'#'Reacher'#'Push'#'PushOutbursts2'#'ReacherInteractions'##'ReacherConstraintdense2'#'ReacherConstraintdense1'#'SimplePointBotConstraints'#modify as needed!
real_dir = os.path.join('', 'data',directory)  #
file=real_dir

utils.modify_trajectories(num_traj,file,horizon)