from src.State import State
from src.base.ContinueActions import ContinueActions
from src.base.GridRewards import GridRewards, GridRewardParams


class RewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        self.data_multiplier = 1.0


# Class used to track rewards
class Rewards(GridRewards):
    cumulative_reward: float = 0.0
    cumulative_reward_distance: float = 0.0
    cumulative_reward_collect: float = 0.0

    def __init__(self, reward_params: RewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

    def calculate_reward(self, state: State, action: ContinueActions, next_state: State):
        reward_moveing = self.calculate_motion_rewards(state, action, next_state)
        self.cumulative_reward_distance += reward_moveing

        # Reward the collected data
        reward_collecting = self.params.data_multiplier * (state.get_remaining_data() - next_state.get_remaining_data())
        self.cumulative_reward_collect+=reward_collecting
        # Cumulative reward
        reward_this_step=reward_moveing+reward_collecting
        self.cumulative_reward += (reward_moveing +reward_collecting)

        return reward_this_step
