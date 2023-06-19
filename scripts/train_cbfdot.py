
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/cuijin/Project6remote/latent-space-safe-sets')

#from latentsafesets.rl_trainers import CBFdotTrainer
from latentsafesets.rl_trainers import CBFdotlatentplanaTrainer#CBFdotTrainer, 
import latentsafesets.utils as utils
#from latentsafesets.utils.arg_parser import parse_args
#from latentsafesets.utils.arg_parser_spb import parse_args
from latentsafesets.utils.arg_parser_reacher import parse_args
#from latentsafesets.utils.arg_parser_push import parse_args
#from latentsafesets.modules import CBFdotEstimator

import os
import logging
import pprint
log = logging.getLogger("main")


if __name__ == '__main__':
    params = parse_args()

    logdir = params['logdir']
    os.makedirs(logdir)
    utils.init_logging(logdir)

    utils.seed(params['seed'])
    log.info('Training cbfdot with params...')
    log.info(pprint.pformat(params))

    env = utils.make_env(params)

    modules = utils.make_modulessafety(params, cbfd=True)
    encoder = modules['enc']#not used if only using states, will be used if using latent states
    cbfdot = modules['cbfd']#CBFdotEstimator(encoder,params)#

    #replay_buffer = utils.load_replay_buffer(params, encoder)
    replay_buffer_success = utils.load_replay_buffer_success(params, encoder)#
    if params['unsafebuffer']=='yes' or params['unsafebuffer']=='yes2':#new version
        replay_buffer_unsafe = utils.load_replay_buffer_unsafe(params, encoder)#around line 123 in utils.py
        log.info('unsafe buffer!')
    else:
        replay_buffer_unsafe=None#replay_buffer
        log.info('the same buffer!')#have checked np.random.randint, it is completely random! This is what I want!
    loss_plotter = utils.LossPlotter(logdir)

    #trainer = CBFdotTrainer(env, params, cbfdot, loss_plotter)
    trainer = CBFdotlatentplanaTrainer(env, params, cbfdot, loss_plotter)
    #trainer.initial_train(replay_buffer, logdir)
    #trainer.initial_train(replay_buffer, logdir,replay_buffer_unsafe)
    if params['ways']==1:
        trainer.initial_train_m2(replay_buffer_success, logdir,replay_buffer_unsafe)
    elif params['ways']==2:
        trainer.initial_train_m2_0109(replay_buffer_success, logdir,replay_buffer_unsafe)
