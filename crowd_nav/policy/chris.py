import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import mlp

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class ForwardDynamicsNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, mlp1_dims, mlp2_dims, mlp3_dims):
        super().__init__()
        self.global_state_dim = mlp1_dims[-1]
        self.action_space = build_action_space()
        self.action_size = len(self.action_space)
        self.action_dim = action_dim
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(self.action_size-1, mlp2_dims, last_relu=True)
        self.mlp3 = mlp(mlp1_dims[-1] + mlp2_dims[-1], mlp3_dims)

    def forward(self, current_state, actions):
        (batch_size, humans, state_dim) = current_state.shape
        mlp1_output = self.mlp1(current_state.view(batch_size * humans, -1))
        mlp2_output = self.mlp2(actions)
        concated = torch.cat([mlp1_output, mlp2_output.repeat(5, 1)], dim=-1)
        mlp3_output = self.mlp3(concated)
        return mlp3_output.view(batch_size, humans, state_dim)


class StatisticsNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp2 = mlp(80, mlp2_dims)
        self.mlp3 = mlp(mlp1_dims[-1] + mlp2_dims[-1], mlp3_dims)
        self.fc = nn.Linear(in_features=5120, out_features=1)
        self.elu = F.elu

    def forward(self, current_state, action):
        (batch_size, humans, state_dim) = current_state.shape
        mlp1_output = self.mlp1(current_state.view(-1, state_dim)) # (500, 61) --> (500, 100)
        mlp1_output = self.elu(mlp1_output)
        mlp2_output = self.mlp2(action) # (100, 2) --> (100, 100)
        mlp2_output = self.elu(mlp2_output)
        concated = torch.cat([mlp1_output, mlp2_output.repeat(5, 1)], dim=-1) # (500, 200)
        x = self.elu(self.mlp3(concated))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class Chris(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'Chris'
        self.action_space = build_action_space().to(self.device)
        self.action_dim = len(self.action_space)
        self.state_dim = 13*5
        self.model = None
        self.target_model = None
        self.with_om = False
        self.multiagent_training = True
        self.last_action_id = None

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.policy_model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)

        self.fwd_model = ForwardDynamicsNetwork(self.input_dim(), self.action_dim, [150, 64], [100, 64],
                                                [128, self.input_dim()])
        self.statistics_model = StatisticsNetwork(self.input_dim(), self.self_state_dim, [150, 100], [150, 100],
                                                  [200, 1024])
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-Chris'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.policy_model.attention_weights

    def get_policy_model(self):
        return self.policy_model

    def get_statistics_model(self):
        return self.statistics_model

    def get_fwd_model(self):
        return self.fwd_model

    def load_weights(self):
        # loads saved networks
        pass

    def value_update(self, state, action, max_action, max_value, reward, rotated_batch_input, action_id, max_action_id):
        next_state_value = self.policy_model(rotated_batch_input).data.item()
        value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
        self.action_values.append(value)
        if value > max_value:
            return action, value, action_id
        else:
            return max_action, max_value, max_action_id

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            build_action_space()

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action_id = np.random.choice(self.action_space.shape[0])
            max_action = self.action_space[max_action_id]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            max_action_id = None
            for action_id, action in enumerate(self.action_space):
                action = ActionXY(action[0], action[1])
                next_self_state = self.propagate(state.self_state, action)
                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                else:
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                       for human_state in state.human_states]
                    reward = self.compute_reward(next_self_state, next_human_states)

                next_human_states = self.obstacle_to_human(next_human_states)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                              for next_human_state in next_human_states if isinstance(next_human_state, ObservableState)], dim=0)
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0).to(self.device)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)
                # VALUE UPDATE
                max_action, max_value, max_action_id = self.value_update(state, action, max_action, max_value, reward, rotated_batch_input, action_id, max_action_id)
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        self.last_action_id = max_action_id
        return max_action


def build_action_space(v_pref=1.0, speed_samples=5, rotation_samples=16):
    """
    Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
    """
    speeds = [(np.exp((i + 1) / speed_samples) - 1) / (np.e - 1) * v_pref for i in
              range(speed_samples)]

    rotations = np.linspace(0, 2 * np.pi, rotation_samples, endpoint=False)
    action_space = [ActionXY(0, 0)]
    for rotation, speed in itertools.product(rotations, speeds):
        action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))

    return torch.FloatTensor(action_space)



