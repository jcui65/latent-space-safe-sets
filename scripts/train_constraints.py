
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')

from latentsafesets.rl_trainers import ConstraintTrainer
import latentsafesets.utils as utils
from latentsafesets.utils.arg_parser import parse_args

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
    log.info('Training gi with params...')
    log.info(pprint.pformat(params))

    env = utils.make_env(params)

    modules = utils.make_modules(params, constr=True)
    encoder = modules['enc']
    constraint = modules['constr']

    replay_buffer = utils.load_replay_buffer(params, encoder)

    loss_plotter = utils.LossPlotter(logdir)

    trainer = ConstraintTrainer(env, params, constraint, loss_plotter)
    trainer.initial_train(replay_buffer, logdir)
