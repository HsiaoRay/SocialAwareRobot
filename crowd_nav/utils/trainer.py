import torch
import torch.nn as nn
import time
import logging
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable


class EmpowermentTrainer(object):
    def __init__(self,
                 memory,
                 policy_model,
                 statistics_model,
                 forward_dynamics_model,
                 device,
                 batch_size,
                 il_learning_rate_q=0.01,
                 il_learning_rate_s=0.01,
                 il_learning_rate_fwd=0.01,
                 rl_learning_rate_q=0.0001,
                 rl_learning_rate_s=0.0001,
                 rl_learning_rate_fwd=0.001,
                 intrinsic_param = 0.0025,
                 n_epochs=50):
        """
        Train the trainable model of a policy
        """
        self.memory = memory
        self.device = device
        self.batch_size = batch_size
        self.il_learning_rate_q = il_learning_rate_q
        self.il_learning_rate_s = il_learning_rate_s
        self.il_learning_rate_fwd = il_learning_rate_fwd
        self.rl_learning_rate_q = rl_learning_rate_q
        self.rl_learning_rate_s = rl_learning_rate_s
        self.rl_learning_rate_fwd = rl_learning_rate_fwd
        self.criterion_q = nn.MSELoss().to(device)
        self.criterion_fwd = nn.MSELoss().to(device)
        self.criterion_statistics = nn.KLDivLoss().to(device)
        self.n_epochs = n_epochs
        self.policy_model = policy_model.to(self.device)
        self.statistics_model = statistics_model.to(self.device)
        self.forward_dynamics_model = forward_dynamics_model.to(self.device)
        self.optimizer_policy_model = None
        self.optimizer_statistics_model = None
        self.optimizer_forward_dynamics_model = None
        self.intrinsic_param = intrinsic_param
        self.data_loader = None

    def set_learning_rate(self, imitation_learning):
        if imitation_learning:
            learning_rate_q = self.il_learning_rate_q
            learning_rate_s = self.il_learning_rate_s
            learning_rate_fwd = self.il_learning_rate_fwd
        else:
            learning_rate_q = self.rl_learning_rate_q
            learning_rate_s = self.rl_learning_rate_s
            learning_rate_fwd = self.rl_learning_rate_fwd

        logging.info('Current learning rate policy model: %f', learning_rate_q)
        logging.info('Current learning rate statistics model: %f', learning_rate_s)
        logging.info('Current learning rate forward dynamics model: %f', learning_rate_fwd)
        self.optimizer_policy_model = optim.Adam(self.policy_model.parameters(), lr=learning_rate_q)
        self.optimizer_statistics_model = optim.Adam(self.statistics_model.parameters(), lr=learning_rate_s)
        self.optimizer_forward_dynamics_model = optim.Adam(self.forward_dynamics_model.parameters(), lr=learning_rate_fwd)

    def optimize_epoch(self, n_epochs):
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)

        for epoch in range(n_epochs):
            start_time = time.time()
            losses_policy = 0

            for data in self.data_loader:
                states, new_states, actions, action_ids, rewards = data
                states = Variable(states).to(self.device)
                rewards = Variable(rewards).to(self.device)

                losses_policy = self.optimize_policy(states=states, rewards=rewards, losses=losses_policy)

            logging.info('Epoch: %f took %f seconds', epoch, time.time() - start_time)

    def optimize_batch(self, num_batches, augment_rewards=True):
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses_policy = 0
        losses_forward = 0
        losses_statistics = 0

        for b in range(num_batches):
            states, new_states, actions, action_ids, rewards = next(iter(self.data_loader))
            states = Variable(states).to(self.device)
            new_states = Variable(new_states).to(self.device)
            actions = Variable(actions).to(self.device)
            rewards = Variable(rewards).to(self.device)
            action_ids = Variable(action_ids).to(self.device)

            losses_forward = self.optimize_forward(states=states, action_ids=action_ids, new_states=new_states, losses=losses_forward)
            losses_statistics, augmented_rewards = self.optimize_statistics(states=states, new_states=new_states,
                                                                            actions=actions, action_ids=action_ids, losses=losses_statistics)
            if augment_rewards:
                for i in range(len(rewards)):
                    rewards[i] += augmented_rewards[i]

            losses_policy = self.optimize_policy(states=states, rewards=rewards, losses=losses_policy)
        average_loss = losses_policy / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss

    def optimize_policy(self, states, rewards, losses):
        self.optimizer_policy_model.zero_grad()
        outputs = self.policy_model(states)
        loss = self.criterion_q(outputs, rewards)
        loss.backward()
        self.optimizer_policy_model.step()
        losses += loss.data.item()
        return losses

    def optimize_forward(self, states, action_ids, new_states, losses):
        self.forward_dynamics_model.zero_grad()
        outputs = self.forward_dynamics_model(states, self.encode_actions(action_ids, self.forward_dynamics_model.action_size).to(self.device))
        loss = self.criterion_fwd(outputs, new_states.squeeze(dim=1))
        loss.backward()
        self.optimizer_forward_dynamics_model.step()
        losses += loss.data.item()
        return losses

    def optimize_statistics(self, states, new_states, actions, action_ids, losses, intrinsic_param=0.010):
        new_state_marginals = []
        for state in states:
            with torch.no_grad():
                action_dim = self.forward_dynamics_model.action_space.shape[0]
                hot_tensor = self.build_eye(action_dim)
                hot_tensor = hot_tensor.type(torch.float32).to(self.device)
                state = state.unsqueeze(dim=0)
                state = state.repeat(hot_tensor.shape[0], 1, 1).to(self.device)
                n_s = self.forward_dynamics_model(state, hot_tensor)
            n_s = n_s.detach()
            n_s = n_s + state
            n_s = torch.mean(n_s, dim=0)
            n_s = torch.unsqueeze(n_s, dim=0)
            new_state_marginals.append(n_s)

        new_state_marginals = tuple(new_state_marginals)
        new_state_marginals = Variable(torch.cat(new_state_marginals), requires_grad=False)
        self.optimizer_statistics_model.zero_grad()
        action_ids = self.encode_actions(action_ids, action_dim)
        action_ids = action_ids.type(torch.float32).to(self.device)
        p_sa = self.statistics_model(new_states, action_ids)
        p_s_a = self.statistics_model(new_state_marginals, action_ids)

        lower_bound = - torch.mean(-p_s_a) - torch.log(torch.mean(torch.exp(p_s_a)))
        mutual_information = - p_sa - torch.log(torch.exp(p_s_a))

        # Maximize the mutual information
        loss = -lower_bound
        loss.backward()
        losses += loss.data.item()
        self.optimizer_statistics_model.step()

        mutual_information = mutual_information.detach()
        mutual_information = torch.clamp(input=mutual_information, min=-1., max=1.)

        augmented_rewards = self.intrinsic_param * mutual_information
        # augmented_rewards = mutual_information
        augmented_rewards.detach()

        return loss, augmented_rewards

    def encode_actions(self, ids, action_dim):
        batch_size = ids.shape[0]
        hot_tensor = torch.zeros(batch_size, action_dim)
        for i, id in enumerate(ids):
            hot_tensor[i][id] = 1
        return hot_tensor[:, 1:]

    def build_eye(self, action_dim):
        eye = torch.eye(action_dim)
        return eye[:, 1:]