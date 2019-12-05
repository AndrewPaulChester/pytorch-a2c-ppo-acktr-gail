import math

import torch

from a2c_ppo_acktr.model import Policy


class WrappedPolicy(Policy):
    def __init__(
        self,
        obs_shape,
        action_space,
        device,
        base=None,
        base_kwargs=None,
        deterministic=False,
        dist=None,
        num_processes=1,
        obs_space=None,
    ):
        super(WrappedPolicy, self).__init__(
            obs_shape, action_space, base, base_kwargs, dist, obs_space
        )
        self.deterministic = deterministic
        self.rnn_hxs = torch.zeros(num_processes, 1)
        self.masks = torch.ones(num_processes, 1)
        self.device = device

    def get_action(self, inputs, rnn_hxs=None, masks=None, valid_envs=None):
        # print(inputs.shape)
        # inputs = torch.from_numpy(inputs).float().to(self.device)

        if rnn_hxs is None:
            rnn_hxs = self.rnn_hxs
        if masks is None:
            masks = self.masks

        value, action, action_log_probs, rnn_hxs, probs = self.act(
            inputs, rnn_hxs, masks, self.deterministic
        )  # Need to be careful about rnn and masks - won't work for recursive

        agent_info = {
            "value": value,
            "probs": action_log_probs,
            "rnn_hxs": rnn_hxs,
            "dist": probs,
        }
        explored = action_log_probs < math.log(0.5)
        # return value, action, action_log_probs, rnn_hxs
        return (action, explored), agent_info

    def reset(self):
        pass

