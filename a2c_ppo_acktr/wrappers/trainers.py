from collections import OrderedDict
import torch

from a2c_ppo_acktr.algo.a2c_acktr import A2C_ACKTR
from a2c_ppo_acktr.algo.ppo import PPO
from a2c_ppo_acktr import utils
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class A2CTrainer(A2C_ACKTR, TorchTrainer):
    def __init__(
        self,
        actor_critic,
        value_loss_coef,
        entropy_coef,
        use_gae,
        gamma,
        gae_lambda,
        use_proper_time_limits,
        lr=None,
        eps=None,
        alpha=None,
        max_grad_norm=None,
        acktr=False,
    ):
        super(A2CTrainer, self).__init__(
            actor_critic,
            value_loss_coef,
            entropy_coef,
            lr,
            eps,
            alpha,
            max_grad_norm,
            acktr,
        )
        # unclear if these are actually used
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits
        self.initial_lr = lr

    def decay_lr(self, epoch, num_epochs):
        utils.update_linear_schedule(self.optimizer, epoch, num_epochs, self.initial_lr)

    def train(self, batch):
        self._num_train_steps += 1
        self.train_from_torch(batch)

    def train_from_torch(self, batch):

        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                batch.obs[-1], batch.recurrent_hidden_states[-1], batch.masks[-1]
            ).detach()

        # compute returns
        batch.compute_returns(
            next_value,
            self.use_gae,
            self.gamma,
            self.gae_lambda,
            self.use_proper_time_limits,
        )
        # update agent - return values are only diagnostic
        value_loss, action_loss, dist_entropy = self.update(batch)
        # TODO: add loss + entropy to eval_statistics.

        # re-initialise experience buffer with current state
        batch.after_update()

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.actor_critic]

    def get_snapshot(self):
        return dict(actor_critic=self.actor_critic)


# TODO: merge A2C + PPO trainers
class PPOTrainer(PPO, TorchTrainer):
    def __init__(
        self,
        actor_critic,
        value_loss_coef,
        entropy_coef,
        use_gae,
        gamma,
        gae_lambda,
        use_proper_time_limits,
        lr=None,
        eps=None,
        clip_param=None,
        ppo_epoch=None,
        num_mini_batch=None,
        max_grad_norm=None,
        acktr=False,
    ):
        super(PPOTrainer, self).__init__(
            actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr,
            eps,
            max_grad_norm,
        )
        # unclear if these are actually used
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits
        self.initial_lr = lr

    def decay_lr(self, epoch, num_epochs):
        utils.update_linear_schedule(self.optimizer, epoch, num_epochs, self.initial_lr)

    def train(self, batch):
        self._num_train_steps += 1
        self.train_from_torch(batch)

    def train_from_torch(self, batch):

        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                batch.obs[-1], batch.recurrent_hidden_states[-1], batch.masks[-1]
            ).detach()

        # compute returns
        batch.compute_returns(
            next_value,
            self.use_gae,
            self.gamma,
            self.gae_lambda,
            self.use_proper_time_limits,
        )
        # update agent - return values are only diagnostic
        value_loss, action_loss, dist_entropy = self.update(batch)
        # TODO: add loss + entropy to eval_statistics.

        # re-initialise experience buffer with current state
        batch.after_update()

        """
        Save some statistics for eval
        """
        print(
            f"value loss {value_loss}, action loss {action_loss}, dist_entropy {dist_entropy}"
        )
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics["Value Loss"] = value_loss
            self.eval_statistics["Action Loss"] = action_loss
            self.eval_statistics["Distribution Entropy"] = dist_entropy

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.actor_critic]

    def get_snapshot(self):
        return dict(actor_critic=self.actor_critic)
