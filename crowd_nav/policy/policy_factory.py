from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.chris import Chris

policy_factory['cadrl'] = CADRL
policy_factory['chris'] = Chris
