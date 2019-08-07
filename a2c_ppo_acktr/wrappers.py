import math
import abc
from collections import OrderedDict

import torch
import gtimer as gt


from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.algo.a2c_acktr import A2C_ACKTR
from a2c_ppo_acktr.algo.ppo import PPO
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.storage import RolloutStorage

from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.external_log import LogPathCollector
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer


class WrappedPolicy(Policy):
    def __init__(
        self,
        obs_shape,
        action_space,
        base=None,
        base_kwargs=None,
        deterministic=False,
        dist=None,
    ):
        super(WrappedPolicy, self).__init__(
            obs_shape, action_space, base, base_kwargs, dist
        )
        self.deterministic = deterministic

    def get_action(self, inputs, rnn, masks):
        value, action, action_log_probs, rnn_hxs = self.act(
            inputs, rnn, masks, self.deterministic
        )  # Need to be careful about rnn and masks - won't work for recursive
        agent_info = {"value": value, "probs": action_log_probs}
        explored = action_log_probs < math.log(0.5)
        return value, action, action_log_probs, rnn_hxs
        # return (action, explored), agent_info

    def reset(self):
        pass


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

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.actor_critic]

    def get_snapshot(self):
        return dict(actor_critic=self.actor_critic)


class RolloutStepCollector(LogPathCollector):
    def __init__(
        self,
        env,
        policy,
        device,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        num_processes=1,
    ):
        super().__init__(
            env, policy, max_num_epoch_paths_saved, render, render_kwargs, num_processes
        )
        self.num_processes = num_processes
        self._rollouts = RolloutStorage(
            max_num_epoch_paths_saved,
            num_processes,
            env.observation_space.shape,
            env.action_space,
            policy.recurrent_hidden_state_size,
        )
        obs = env.reset()
        self._rollouts.obs[0].copy_(obs)
        self._rollouts.to(device)

    def get_rollouts(self):
        return self._rollouts

    def collect_one_step(self, step):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = self._policy.get_action(
                self._rollouts.obs[step],
                self._rollouts.recurrent_hidden_states[step],
                self._rollouts.masks[step],
            )

        # Observe reward and next obs
        obs, reward, done, infos = self._env.step(action)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
        )
        gt.stamp("exploration sampling", unique=False)

        self._rollouts.insert(
            obs,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            reward,
            masks,
            bad_masks,
        )
        self.add_step(action, action_log_prob, reward, done, value)


class IkostrikovRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: RolloutStepCollector,
        evaluation_data_collector: RolloutStepCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        use_linear_lr_decay,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.use_linear_lr_decay = use_linear_lr_decay

        assert (
            self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop
        ), "Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop"

    def _train(self):
        self.training_mode(False)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs), save_itrs=True
        ):

            for step in range(self.num_eval_steps_per_epoch):
                self.eval_data_collector.collect_one_step(step)
            gt.stamp("evaluation sampling")

            for _ in range(self.num_train_loops_per_epoch):
                # this if check could be moved inside the function
                if self.use_linear_lr_decay:
                    # decrease learning rate linearly
                    self.trainer.decay_lr(epoch, self.num_epochs)

                for step in range(self.num_expl_steps_per_train_loop):
                    self.expl_data_collector.collect_one_step(step)
                    gt.stamp("data storing", unique=False)

                rollouts = self.expl_data_collector.get_rollouts()
                self.training_mode(True)
                self.trainer.train(rollouts)
                gt.stamp("training", unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)


class TorchIkostrikovRLAlgorithm(IkostrikovRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
