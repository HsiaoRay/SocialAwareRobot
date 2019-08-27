import numpy as np
from numpy.linalg import norm
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import ObservableState, FullState, JointState, Shape


class Dog(Agent):
    def __init__(self):
        """
        Dog behaves unpredictable: it starts moving quickly after some n seconds

        """
        self.v_pref = 1.5
        self.radius = 0.2
        self.policy = policy_factory['orca']()
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = 100.0
        self.py = 100.0
        self.gx = 100.0
        self.gy = 100.0
        self.vx = 0.
        self.vy = 0.
        self.theta = None
        self.time_step = None
        self.t_offset = 0

    def set_t_offset(self, t_offset):
        self.t_offset = t_offset

    def compute_position(self, action, delta_t):
        px = self.px + action.vx * delta_t
        py = self.py + action.vy * delta_t
        return px, py

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        ob = self.observe(ob)
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def step(self, action, time):
        """
        Perform an action and update the state
        """
        if time > self.t_offset:
            pos = self.compute_position(action, self.time_step)
            self.px, self.py = pos
            self.vx = action.vx
            self.vy = action.vy


