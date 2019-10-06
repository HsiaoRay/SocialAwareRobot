import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.action import ActionXY
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
    def __init__(self, input_dim, action_dim, num_human):
        super().__init__()
        self.action_dim = action_dim
        self.input_dim = input_dim * num_human
        self.state_layer1 = nn.Linear(in_features=self.input_dim, out_features=150)
        self.state_layer2 = nn.Linear(in_features=150, out_features=250)
        self.state_layer3 = nn.Linear(in_features=250, out_features=150)
        self.action_layer1 = nn.Linear(in_features=self.action_dim, out_features=150)
        self.action_layer2 = nn.Linear(in_features=150, out_features=250)
        self.action_layer3 = nn.Linear(in_features=250, out_features=150)
        self.concated_layer = nn.Linear(in_features=300, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=self.input_dim)
        self.elu = F.elu

    def forward(self, current_states, actions):
        (batch_size, humans, state_dim) = current_states.shape
        current_states = current_states.view(batch_size, -1)
        current_states = self.state_layer1(current_states)
        current_states = self.elu(current_states)
        current_states = self.state_layer2(current_states)
        current_states = self.elu(current_states)
        current_states = self.state_layer3(current_states)
        current_states = self.elu(current_states)
        actions = actions.view(batch_size, -1)
        actions = self.action_layer1(actions)
        actions = self.elu(actions)
        actions = self.action_layer2(actions)
        actions = self.elu(actions)
        actions = self.action_layer3(actions)
        actions = self.elu(actions)
        concated = torch.cat((current_states, actions), dim=1)
        x = self.concated_layer(concated)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x.view(batch_size, humans, -1)


class StatisticsNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, num_human):
        super().__init__()
        self.action_dim = action_dim
        self.input_dim = input_dim * num_human
        self.state_layer1 = nn.Linear(in_features=self.input_dim, out_features=150)
        self.state_layer2 = nn.Linear(in_features=150, out_features=250)
        self.state_layer3 = nn.Linear(in_features=250, out_features=150)
        self.action_layer1 = nn.Linear(in_features=action_dim, out_features=150)
        self.action_layer2 = nn.Linear(in_features=150, out_features=250)
        self.action_layer3 = nn.Linear(in_features=250, out_features=150)
        self.concated_layer = nn.Linear(in_features=300, out_features=512)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)
        self.relu = F.relu
        self.elu = F.elu

    def forward(self, current_states, actions):
        (batch_size, humans, state_dim) = current_states.shape
        current_states = current_states.view(batch_size, -1)
        current_states = self.state_layer1(current_states)
        current_states = self.relu(current_states)
        current_states = self.state_layer2(current_states)
        current_states = self.relu(current_states)
        current_states = self.state_layer3(current_states)
        current_states = self.relu(current_states)
        actions = actions.view(batch_size, -1)
        actions = self.action_layer1(actions)
        actions = self.relu(actions)
        actions = self.action_layer2(actions)
        actions = self.relu(actions)
        actions = self.action_layer3(actions)
        actions = self.relu(actions)
        concated = torch.cat((current_states, actions), dim=1)
        x = self.concated_layer(concated)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.elu(x)


class Chris(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'Chris'
        self.action_space = build_action_space().to(self.device)
        self.action_dim = len(self.action_space)-1
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
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-Chris'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def make_fwd_model(self, num_hum):
        self.fwd_model = ForwardDynamicsNetwork(self.input_dim(), self.action_dim, num_hum)

    def make_stats_model(self, num_hum):
        self.statistics_model = StatisticsNetwork(self.input_dim(), self.action_dim, num_hum)

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

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
        if self.with_om:
            self_state = np.array([state.self_state.px, state.self_state.py, state.self_state.vx,
                                   state.self_state.vx])
            occupancy_maps = self.build_state_space(self_state, state.human_states).to(self.device)
            #occupancy_maps = self.build_occupancy_maps(state.human_states).to(self.device)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
        return state_tensor


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
            return ActionXY(0, 0)
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
                        self_state = np.array([next_self_state.px.numpy(), next_self_state.py.numpy(), next_self_state.vx.numpy(), next_self_state.vx.numpy()])
                        occupancy_maps = self.build_state_space(self_state, next_human_states).unsqueeze(0).to(self.device)
                        #occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0).to(self.device)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)
                # VALUE UPDATE
                max_action, max_value, max_action_id = self.value_update(state, action, max_action, max_value, reward, rotated_batch_input, action_id, max_action_id)
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        self.last_action_id = max_action_id
        return max_action

    def build_state_space(self, self_state, human_states):
        """

         :param human_states:
         :return: tensor of shape (# human - 1, self.cell_num ** 2)
         """
        occupancy_maps = []
        other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                       for other_human in human_states], axis=0)
        other_px = other_humans[:, 0] - self_state[0]
        other_py = other_humans[:, 1] - self_state[1]
        # new x-axis is in the direction of human's velocity
        human_velocity_angle = np.arctan2(self_state[3], self_state[2])
        other_human_orientation = np.arctan2(other_py, other_px)
        rotation = other_human_orientation - human_velocity_angle
        distance = np.linalg.norm([other_px, other_py], axis=0)
        other_px = np.cos(rotation) * distance
        other_py = np.sin(rotation) * distance

        # compute indices of humans in the grid
        other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
        other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
        other_x_index[other_x_index < 0] = float('-inf')
        other_x_index[other_x_index >= self.cell_num] = float('-inf')
        other_y_index[other_y_index < 0] = float('-inf')
        other_y_index[other_y_index >= self.cell_num] = float('-inf')
        grid_indices = self.cell_num * other_y_index + other_x_index
        occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
        # self.visualize_om(occupancy_map)
        if self.om_channel_size == 1:
            occupancy_maps.append([occupancy_map.astype(int)])
        else:
            # calculate relative velocity for other agents
            other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
            rotation = other_human_velocity_angles - human_velocity_angle
            speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
            other_vx = np.cos(rotation) * speed
            other_vy = np.sin(rotation) * speed

            for i, index in np.ndenumerate(grid_indices):
                dm = np.zeros(self.cell_num ** 2 * self.om_channel_size)
                if index in range(self.cell_num ** 2):
                    if self.om_channel_size == 2:
                        dm[2 * int(index)] = other_vx[i]
                        dm[2 * int(index) + 1] = other_vy[i]
                    elif self.om_channel_size == 3:
                        dm[3 * int(index)] = 1
                        dm[3 * int(index) + 1] = other_vx[i]
                        dm[3 * int(index) + 2] = other_vy[i]
                    else:
                        raise NotImplementedError

                # for i, cell in enumerate(dm):
                #     dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        # self.visualize_dm(torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float())
        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

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






