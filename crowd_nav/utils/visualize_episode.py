import logging
import numpy as np
import parser

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.info import *


def visualize_episode(env, robot, args):
    ob = env.reset(phase='test', test_case=args.test_case)
    done = False
    last_pos = np.array(robot.get_position())
    while not done:
        action = robot.act(ob)
        action = ActionXY(action[0], action[1])
        ob, _, done, info = env.step(action)
        current_pos = np.array(robot.get_position())
        logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
        last_pos = current_pos

    logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
    if robot.visible and isinstance(info, ReachGoal):
        (human_times, human_distances) = env.get_human_times()
        logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))

    if (args.vis_type == 'traj' and args.output_file[-4:] == ".png") or (args.vis_type == 'video' and args.output_file[-4:] == ".mp4"):
        env.render(mode=args.vis_type, output_file=args.output_file)
    elif args.vis_type == 'traj':
        env.render(mode=args.vis_type)
