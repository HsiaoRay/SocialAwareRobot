
import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import git

from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import EmpowermentTrainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import EmpowermentExplorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.visualize_episode import visualize_episode


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='chris')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--output_dir', type=str, default='data/output_empowerment')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--output_file', type=str, default='data/')
    key = input('Dubug mode? (y/n)')
    if key == 'y':
        parser.add_argument('--train_config', type=str, default='configs/debug.config')
    else:
        parser.add_argument('--train_config', type=str, default='configs/train.config')
    args = parser.parse_args()

    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')
    il_policy_model_weight_file = os.path.join(args.output_dir, 'il_policy_model.pth')
    rl_policy_model_weight_file = os.path.join(args.output_dir, 'rl_policy_model.pth')
    il_statistics_model_weight_file = os.path.join(args.output_dir, 'il_statistics_model.pth')
    rl_statistics_model_weight_file = os.path.join(args.output_dir, 'rl_statistics_model.pth')
    il_forward_dynamics_model_weight_file = os.path.join(args.output_dir, 'il_forward_dynamics_model.pth')
    rl_forward_dynamics_model_weight_file = os.path.join(args.output_dir, 'rl_forward_dynamics_model.pth')

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: %s'.format(repo.head.object.hexsha))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    if args.policy_config is None:
        parser.error('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)
    policy.set_device(device)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # read training parameters
    if args.train_config is None:
        parser.error('Train config has to be specified for a trainable network')
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    train_batches = train_config.getint('train', 'train_batches')
    train_episodes = train_config.getint('train', 'train_episodes')
    sample_episodes = train_config.getint('train', 'sample_episodes')
    target_update_interval = train_config.getint('train', 'target_update_interval')
    evaluation_interval = train_config.getint('train', 'evaluation_interval')
    capacity = train_config.getint('train', 'capacity')
    epsilon_start = train_config.getfloat('train', 'epsilon_start')
    epsilon_end = train_config.getfloat('train', 'epsilon_end')
    epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
    checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

    # configure trainer and explorer
    memory = ReplayMemory(capacity)

    batch_size = train_config.getint('trainer', 'batch_size')
    trainer = EmpowermentTrainer(memory=memory,
                                 policy_model=policy.get_policy_model(),
                                 statistics_model=policy.get_statistics_model(),
                                 forward_dynamics_model=policy.get_fwd_model(),
                                 batch_size=batch_size,
                                 device=device)

    policy_model = policy.get_policy_model()
    statistics_model = policy.get_statistics_model()
    forward_dynamics_model = policy.get_fwd_model()

    explorer = EmpowermentExplorer(env, robot, device, memory, .9, target_policy=policy)

    # imitation learning
    if args.resume:
        if os.path.exists(rl_policy_model_weight_file):
            policy_model.load_state_dict(torch.load(rl_policy_model_weight_file))
            rl_policy_model_weight_file = os.path.join(args.output_dir, 'resumed_rl_policy_model.pth')
            logging.info('Load reinforcement learning trained weights for Policy Model. Resume training')
        else:
            logging.error('Weights for Policy Model do not exist!')

        if os.path.exists(rl_statistics_model_weight_file):
            statistics_model.load_state_dict(torch.load(rl_statistics_model_weight_file))
            rl_statistics_model_weight_file = os.path.join(args.output_dir, 'resumed_rl_statistics_model.pth')
            logging.info('Load reinforcement learning trained weights for Statistics Model. Resume training')
        else:
            logging.error('Weights for Statistics Model do not exist!')

        if os.path.exists(rl_forward_dynamics_model_weight_file):
            forward_dynamics_model.load_state_dict(torch.load(rl_forward_dynamics_model_weight_file))
            rl_forward_dynamics_model_weight_file = os.path.join(args.output_dir, 'rl_forward_dynamics_model.pth')
            logging.info('Load reinforcement learning trained weights for Policy Model. Resume training')
        else:
            logging.error('Weights for Forward Dynamics Model do not exist!')

    elif os.path.exists(il_policy_model_weight_file) \
            and os.path.exists(il_statistics_model_weight_file) \
            and os.path.exists(il_forward_dynamics_model_weight_file):
        policy_model.load_state_dict(torch.load(il_policy_model_weight_file))
        statistics_model.load_state_dict(torch.load(il_statistics_model_weight_file))
        forward_dynamics_model.load_state_dict(torch.load(il_forward_dynamics_model_weight_file))
        logging.info('Load imitation learning trained weights for policy, statistics and dynamic forward model.')
    else:
        il_episodes = train_config.getint('imitation_learning', 'il_episodes')
        il_policy = train_config.get('imitation_learning', 'il_policy')
        il_epochs = train_config.getint('imitation_learning', 'il_epochs')
        trainer.set_learning_rate(imitation_learning=True)
        if robot.visible:
            safety_space = 0.0
        else:
            safety_space = train_config.getfloat('imitation_learning', 'safety_space')
        il_policy = policy_factory[il_policy]()
        if il_policy:
            il_policy.multiagent_training = policy.multiagent_training
            il_policy.safety_space = safety_space
            robot.set_policy(il_policy)
            explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)

        trainer.optimize_epoch(il_epochs)
        torch.save(policy_model.state_dict(), il_policy_model_weight_file)
        torch.save(statistics_model.state_dict(), il_statistics_model_weight_file)
        torch.save(forward_dynamics_model.state_dict(), il_forward_dynamics_model_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    explorer.update_target_model(policy_model)

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_learning_rate(imitation_learning=False)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(1, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # evaluate the model
        if episode % evaluation_interval == 0 and episode != 0:
            explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        trainer.optimize_batch(train_batches, augment_rewards=True)
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(policy_model)

        if episode != 0 and episode % checkpoint_interval == 0:
            args.vis_type = 'traj'
            args.test_case = 12
            args.output_file = 'episode_{}.png'.format(episode)
            robot.policy.set_phase('test')
            visualize_episode(robot=robot, env=env, args=args)
            robot.policy.set_phase('train')

            torch.save(policy_model.state_dict(), rl_policy_model_weight_file)
            torch.save(statistics_model.state_dict(), rl_statistics_model_weight_file)
            torch.save(forward_dynamics_model.state_dict(), rl_forward_dynamics_model_weight_file)

    # final test
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)


if __name__ == '__main__':
    main()
