import random
import numpy as np

class StudentEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.fuel = 100
        self.attendance = 100
        self.speed = 0
        self.distance = 0
        self.time = 0

        return self._get_state()

    def _get_state(self):
        return np.array([
            self.fuel / 100,
            self.attendance / 100,
            self.speed / 10,
            self.time / 100
        ])

    def step(self, action):
        reward = 0

        # Actions
        if action == 0:  # accelerate
            self.speed += 1
            self.fuel -= 5
            self.distance += self.speed

        elif action == 1:  # slow down
            self.speed = max(0, self.speed - 1)
            self.fuel -= 2

        elif action == 2:  # break
            self.fuel += 5
            self.speed = max(0, self.speed - 1)

        elif action == 3:  # attend class
            self.attendance += 2
            self.fuel -= 2

        elif action == 4:  # skip class
            self.attendance -= 5
            self.distance += 2

        # Clamp values
        self.fuel = max(0, min(100, self.fuel))
        self.attendance = max(0, min(100, self.attendance))

        self.time += 1

        # Reward function
        reward = (
    self.distance * 0.01 +
    self.attendance * 0.02 -
    (100 - self.fuel) * 0.015
)
        # Done conditions
        done = False
        if self.fuel <= 0 or self.attendance < 75 or self.time > 200:
            done = True

        return self._get_state(), reward, done