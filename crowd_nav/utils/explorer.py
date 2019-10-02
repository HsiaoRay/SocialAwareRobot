import logging
import copy
import torch
import numpy as np
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.utils.visualize_episode import visualize_episode

class EmpowermentExplorer(object):
    def __init__(self, env, robot, device, args, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        self.stats = None
        self.args = args

    def update_target_model(self, model):
        self.target_model = copy.deepcopy(model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        nav_distances = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        hum_travel_dist = []
        hum_travel_time = []

        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            action_ids = []
            while not done:
                action = self.robot.act(ob)
                action = ActionXY(action[0], action[1])
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                if imitation_learning:
                    action_ids.append(0)
                else:
                    action_ids.append(self.robot.policy.last_action_id)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)


            if i % 100 == 0:
                self.args.vis_type = 'traj'
                self.args.test_case = 12
                self.args.output_file = 'episode_{}.png'.format(i)
                self.robot.policy.set_phase('test')
                visualize_episode(robot=self.robot, env=self.env, args=self.args)
                self.robot.policy.set_phase('train')

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)

                if phase in ['test', 'val']:
                    nav_distances.append(sum([(action.vx ** 2 + action.vy ** 2) ** .5 * self.robot.time_step for action in actions]))
                    (human_times, human_distances) = self.env.get_human_times()
                    hum_travel_dist.append(average(human_distances))
                    hum_travel_time.append(average(human_times))

            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, action_ids, rewards, imitation_learning)

            cumulative_reward = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)])
            cumulative_rewards.append(cumulative_reward)

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        test_info = '' if phase not in ['test', 'val'] else ', nav distance: {},  human distance: {:.2f}, human travel time: {:.2f}'.format(average(nav_distances), average(hum_travel_dist), average(hum_travel_time))
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.3f}, '
                     'nav time: {:.2f}, total reward: {:.4f} {}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards), test_info))

    def update_memory(self, states, actions, action_ids, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]
            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
                if i == len(states) - 1:
                    next_state = states[i]
                else:
                    next_state = states[i + 1]
                next_state = self.target_policy.transform(next_state)
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                    next_state = states[i]
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()

            action = torch.Tensor([actions[i][0], actions[i][1]]).to(self.device)
            value = torch.Tensor([value]).to(self.device)

            self.memory.push((state, next_state, action, action_ids[i], value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0