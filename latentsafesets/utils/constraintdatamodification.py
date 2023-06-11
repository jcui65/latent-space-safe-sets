import latentsafesets.utils as utils
import os
num_traj=300#50#100#
horizon=150
directory='PushOutbursts2'#'ReacherInteractions'##'ReacherConstraintdense2'#'ReacherConstraintdense1'#'SimplePointBotConstraints'#modify as needed!
real_dir = os.path.join('', 'data',directory)  #
file=real_dir

utils.modify_trajectories(num_traj,file,horizon)