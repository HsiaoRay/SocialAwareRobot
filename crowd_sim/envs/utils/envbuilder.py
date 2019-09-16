import numpy as np
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.dog import Dog
from crowd_sim.envs.utils.obstacle import Obstacle

class EnvBuilder():
    def __init__(self, config, test_case):
        self.humans = []
        self.dogs = []
        self.obstacles = []
        self.config = config
        self.test_case = test_case

    def set_humans(self):
        if self.test_case == -1:
            human_settings = [
                # activate, px, py, gx, gy, vx, vy
                [True,  3.5, 0.0, -3.5, 0.0, 0.0, 0.0, 0.0],
                [True, 4.5, 1.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                [True, 4.5, 2, -4.0, 2, 0.0, 0.0, 0.0],
                [True, 4.5, 3.0, -4.0, 3.0, 0.0, 0.0, 0.0],
                [False, 5.0, 3.0, -6.5, 3.0, 0.0, 0.0, 0.0],
            ]

        if self.test_case == -2:
            radius = 1.0
            human_settings = []
            for i_human in range(5):
                angle = 2 * np.pi / 5 * i_human
                setting = [True, radius * np.sin(angle), radius * np.cos(angle), radius * np.sin(angle), radius * np.cos(angle), 0.0, 0.0, 0.0]
                human_settings.append(setting)

        if self.test_case == -3:
            human_settings = [
                # activate, px, py, gx, gy, vx, vy
                [True,  0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [True, -1.5, 0.0, 8.5, 0.0, 0.0, 0.0, 0.0],
                [False, 4.5, 2, -4.0, 2, 0.0, 0.0, 0.0],
                [False, 4.5, 3.0, -4.0, 3.0, 0.0, 0.0, 0.0],
                [False, 5.0, 3.0, -6.5, 3.0, 0.0, 0.0, 0.0],
            ]

        for human_setting in human_settings:
            if human_setting[0]:
                human = Human(self.config, 'humans')
                px = human_setting[1]
                py = human_setting[2]
                gx = human_setting[3]
                gy = human_setting[4]
                vx = human_setting[5]
                vy = human_setting[6]
                theta = human_setting[7]
                human.set(px=px, py=py, gx=gx, gy=gy, vx=vx, vy=vy, theta=theta)
                self.humans.append(human)

    def set_dogs(self):
        dog_settings = [
            # activate, px, py, gx, gy, vx, vy, theta, time_offset
            [False, 2.0, -3.5, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0],
            [False, 2.0, 2.0, 0.0, -5.0, 0.0, 0.0, 0.0, 2.0],
            [False, 0.0, -10.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0],
            [False, 0.0, -10.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0],
            [False, 0.0, -10.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0,0]
        ]

        for dog_setting in dog_settings:
            if dog_setting[0]:
                dog = Dog()
                px = dog_setting[1]
                py = dog_setting[2]
                gx = dog_setting[3]
                gy = dog_setting[4]
                vx = dog_setting[5]
                vy = dog_setting[6]
                theta = dog_setting[7]
                t_offset = dog_setting[8]
                dog.set(px=px, py=py, gx=gx, gy=gy, vx=vx, vy=vy, theta=theta)
                dog.set_t_offset(t_offset)
                self.dogs.append(dog)

    def set_obstacles(self):
        if self.test_case == -1:
            obstacle_settings = [
                # activate, px, py
                [True, 0, 0],
                [False, 0, -10],
                [False, 0, -10],
                [False, 0, -10],
                [False, 0, -10]
            ]

            for obstacle_setting in obstacle_settings:
                if obstacle_setting[0]:
                    obstacle = Obstacle()
                    px = obstacle_setting[1]
                    py = obstacle_setting[2]
                    obstacle.set_position((px, py))
                    self.obstacles.append(obstacle)

    def get_data(self):
        self.set_humans()
        self.set_dogs()
        self.set_obstacles()
        return self.humans, self.dogs, self.obstacles
