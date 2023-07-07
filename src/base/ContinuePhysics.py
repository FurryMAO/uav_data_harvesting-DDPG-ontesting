from src.base.ContinueActions import ContinueAction
import numpy as np


class ContinuePhysics:
    def __init__(self):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = None

    def movement_step(self, action: ContinueAction):
        old_position = self.state.position
        x, y = old_position
        x += action.dx
        y += action.dy
        self.state.set_position([x, y])
        if action.dx==0 and action.dy==0:
            self.landing_attempts += 1
            if self.state.is_in_landing_zone():
                self.state.set_landed(True)

        if self.state.is_in_no_fly_zone():
            # Reset state
            self.boundary_counter += 1
            x, y = old_position
            self.state.set_position([x, y])

        self.state.decrement_movement_budget(action.dx,action.dy)
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))

        return x, y

    def reset(self, state):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state