from crowd_sim.envs.utils.state import Shape


class Obstacle(object):
    def __init__(self):
        self.px = 0
        self.py = 0
        self.radius = 0.5

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_shape(self):
        return Shape(self.px, self.py, self.radius)