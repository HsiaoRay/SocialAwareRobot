import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import EmpowermentExplorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.info import *
from crowd_nav.utils.plot import distribution_humans


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--vis_type', type=str, default='snapshots')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            policy_model_weight_file = os.path.join(args.model_dir, 'il_policy_model.pth')
            statistics_model_weight_file = os.path.join(args.model_dir, 'il_statistics_model.pth')
            forward_dynamics_model_weight_file = os.path.join(args.model_dir, 'il_forward_dynamics_model.pth')
        else:
            policy_model_weight_file = os.path.join(args.model_dir, 'rl_policy_model.pth')
            statistics_model_weight_file = os.path.join(args.model_dir, 'rl_statistics_model.pth')
            forward_dynamics_model_weight_file = os.path.join(args.model_dir, 'rl_forward_dynamics_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_policy_model().load_state_dict(torch.load(policy_model_weight_file, map_location=device))
        policy.get_statistics_model().load_state_dict(torch.load(statistics_model_weight_file, map_location=device))
        policy.get_fwd_model().load_state_dict(torch.load(forward_dynamics_model_weight_file, map_location=device))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = EmpowermentExplorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    if args.visualize:
        if args.vis_type in ['traj', 'video', 'snapshots']:
            ob = env.reset(args.phase, args.test_case)
            done = False
            last_pos = np.array(robot.get_position())
            while not done:
                action = robot.act(ob)
                action = ActionXY(action[0], action[1])
                ob, _, done, info = env.step(action)
                current_pos = np.array(robot.get_position())
                logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
                last_pos = current_pos
            env.render(args.vis_type, args.video_file)
            logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
            if robot.visible and isinstance(info, ReachGoal):
                human_times = env.get_human_times()
                logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
        elif args.vis_type == '2d_histogram' or args.vis_type == 'distance_distribution':
            env.discomfort_dist = .5
            n_tests = 100
            n_reached_goal = 0
            n_too_close = 0
            min_dist = []

            for test_num in range(n_tests):
                ob = env.reset(args.phase, test_num)
                done = False
                last_pos = np.array(robot.get_position())
                while not done:
                    action = robot.act(ob)
                    action = ActionXY(action[0], action[1])
                    ob, _, done, info = env.step(action)
                    current_pos = np.array(robot.get_position())
                    logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
                    last_pos = current_pos
                if isinstance(info, ReachGoal):
                    n_reached_goal += 1

                env.render_k_tests(test_num, n_tests)

                if isinstance(info, Danger):
                    print(info.min_dist)
                    n_too_close += 1
                    min_dist.append(info.min_dist)

                if n_reached_goal != 0 and test_num != 0:
                    logging.info('It takes %.2f seconds to finish. Final status is %s. Progress is %.f%%. Success rate is %.f%%.', env.global_time, info, ((test_num - 1 )/ n_tests) * 100,
                                 (n_reached_goal / test_num - 1) * 100)
            distribution_humans(min_dist, n_tests)

    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()
