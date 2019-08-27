# SocialAwareRobot
Our robot, called Chris, moves among humans and obstacles. It is driven by empowerment to make it social aware.

# What is social intelligence?
- Letting humans pass
- Don't block the way of humans
- Don't change the path of humans
- Don't push humans
- Reach goal in a realistic time!

# How do you achieve this?
We employ three networks:
- The Q-network takes actions that lead to empowered states.
- The forward dynamics network is used to predict a future state from the current state and an action.
- The statistics network is used to generate the mutual information.

## Folder structure
The folders are organized as follows:
- [crowd_nav](crowd_nav) contains scripts for training and testing.
- [crowd_nav/policy](crowd_nav/policy) contains the RL agent class (chris.py). This script also contains the q-, statistics and forward dynamics networks.
- [crowd_sim/envs](crowd_sim/envs) contains the environment (crowd_sim.py).

A reward of +1 is provided if the robot reaches his goal. A reward of -.25 is provided if he collides with a human or obstacle. 

The state space has 8 dimensions for the robot (px, py, radius, vx, vy, gx, gy, theta, v_pref) and 5 for the humans (px, py, radius, vx, vy). The action space is 2 (vx, vy).

## Getting started
If you want to train a policy use the following command:
```
python train.py --policy chris
```

If you want to test a policy, you can use the following command:
```
python test.py --policy chris --model_dir data/output --phase test --visualize --test_case 0
```

# Result
The first video shows the robot after it has been trained without empowerment. The second video shows the robot that does not affect the human's movements.
![](readme/bad.gif)

![](readme/good.gif)

# Architecture 

![](readme/architecture.png)



