from src.base.ContinueActions import ContinueActions
import numpy as np
class ContinuePhysics:
    def __init__(self):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = None

    def movement_step(self, old_position, action):
        x = old_position[0]
        y= old_position[1]
        #print('the action is', action)
        dx=np.cos(action)
        dy=-np.sin(action)
        #print('the movemnet is:',(dx,dy))

        x_new=x+dx
        y_new=y+dy
        self.state.set_position([x_new, y_new])

        if self.state.is_in_landing_zone():
            #print('in landing area')
            self.landing_attempts += 1
            if self.state.movement_budget<=25:
                self.state.set_landed(True)

        if self.state.is_in_no_fly_zone():
            #print('fly into no fly area, can not move')
            # Reset state
            self.boundary_counter += 1
            self.state.set_position([x, y])


        self.state.decrement_movement_budget()
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))
        # print(old_position)
        # print(x,y)
        return self.state.position

    def reset(self, state):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state
