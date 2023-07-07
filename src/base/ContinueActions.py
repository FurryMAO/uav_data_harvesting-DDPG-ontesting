class ContinueAction:
    def __init__(self, input):
        max_velocity=5
        dx=input[0]
        dy=input[1]
        self.dx = min(dx, max_velocity)
        self.dy = min(dy, max_velocity)