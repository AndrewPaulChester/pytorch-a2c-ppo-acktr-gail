import math
import abc
from collections import OrderedDict
import numpy as np

from gym.spaces import Tuple
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
import rlkit.torch.pytorch_util as ptu
from rlkit.util.multi_queue import MultiQueue

from gym_taxi.utils.spaces import Json


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

        value, action, action_log_probs, rnn_hxs = self.act(
            inputs, rnn_hxs, masks, self.deterministic
        )  # Need to be careful about rnn and masks - won't work for recursive

        agent_info = {"value": value, "probs": action_log_probs, "rnn_hxs": rnn_hxs}
        explored = action_log_probs < math.log(0.5)
        # return value, action, action_log_probs, rnn_hxs
        return (action, explored), agent_info

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


def _flatten_tuple(observation):
    """Assumes observation is a tuple of tensors. converts ((n,c, h, w),(n, x)) -> (n,c*h*w+x)"""
    image, fc = observation
    flat = image.view(image.shape[0], -1)
    return torch.cat((flat, fc), dim=1)


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

        self.device = device
        self.is_json = isinstance(env.observation_space, Json)
        self.is_tuple = False
        if self.is_json:
            self.json_to_screen = env.observation_space.converter
            self.is_tuple = isinstance(env.observation_space.image, Tuple)

        shape = (
            (
                (
                    env.observation_space.image[0].shape,
                    env.observation_space.image[1].shape,
                )
                if self.is_tuple
                else env.observation_space.image.shape
            )
            if self.is_json
            else env.observation_space.shape
        )
        self._rollouts = RolloutStorage(
            max_num_epoch_paths_saved,
            num_processes,
            shape,
            env.action_space,
            1,  # hardcoding reccurent hidden state off for now.
        )

        raw_obs = env.reset()
        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs
        self.obs = (
            raw_obs if isinstance(self, HierarchicalStepCollector) else action_obs
        )

        # print(raw_obs.shape)
        # print(action_obs.shape)
        # print(stored_obs.shape)

        self._rollouts.obs[0].copy_(stored_obs)
        self._rollouts.to(self.device)

    def _convert_to_torch(self, raw_obs):
        if self.is_json:
            list_of_observations = [self.json_to_screen(o[0]) for o in raw_obs]
            if self.is_tuple:
                tuple_of_observation_lists = zip(*list_of_observations)
                action_obs = tuple(
                    [
                        torch.tensor(list_of_observations).float().to(self.device)
                        for list_of_observations in tuple_of_observation_lists
                    ]
                )
            else:
                action_obs = torch.tensor(list_of_observations).float().to(self.device)
        else:
            action_obs = torch.tensor(raw_obs).float().to(self.device)
        return action_obs

    def get_rollouts(self):
        return self._rollouts

    def collect_one_step(self, step, step_total):
        with torch.no_grad():
            (action, explored), agent_info = self._policy.get_action(self.obs)

        value = agent_info["value"]
        action_log_prob = agent_info["probs"]
        recurrent_hidden_states = agent_info["rnn_hxs"]

        # Observe reward and next obs
        raw_obs, reward, done, infos = self._env.step(ptu.get_numpy(action))
        if self._render:
            self._env.render(**self._render_kwargs)

        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs
        self.obs = action_obs

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
        )
        gt.stamp("exploration sampling", unique=False)

        self._rollouts.insert(
            stored_obs,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            reward,
            masks,
            bad_masks,
        )
        self.add_step(action, action_log_prob, reward, done, value)


class HierarchicalStepCollector(RolloutStepCollector):
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
            env,
            policy,
            device,
            max_num_epoch_paths_saved,
            render,
            render_kwargs,
            num_processes,
        )
        self.action_queue = MultiQueue(num_processes)
        self.obs_queue = MultiQueue(num_processes)
        self.cumulative_reward = np.zeros(num_processes)

    def collect_one_step(self, step, step_total):
        """
        This needs to handle the fact that different environments can have different plan lengths, and so the macro steps are not in sync. 

        Issues: 
            cumulative reward
            multiple queued experiences for some environments
            identifying termination

        While there is an environment which hasn't completed one macro-step, it takes simultaneous steps in all environments.
        Keeps two queues, action and observation, which correspond to the start and end of the macro step.
        Each environment is checked to see if the current information should be added to the action or observation queues.
        When the observation queue (which is always 0-1 steps behind the action queue) has an action for all environments, these
        are retrieved and inserted in the rollout storage.

        To ensure that environments stay on policy (i.e. we don't queue up lots of old experience in the buffer), and we don't plan more 
        than required, each step we check to see if any environments have enough experience for this epoch, and if so we execute a no-op.
        """
        remaining = step_total - step

        step_count = 0
        while not self.obs_queue.check_layer():
            # print(remaining, step_total, step)
            valid_envs = [len(q) < remaining for q in self.obs_queue.queues]
            # print(valid_envs)
            with torch.no_grad():
                results = self._policy.get_action(self.obs, valid_envs=valid_envs)
                # print(results)
            action = np.array([[a] for (a, _), _ in results])

            # Observe reward and next obs
            raw_obs, reward, done, infos = self._env.step(action)
            if self._render:
                self._env.render(**self._render_kwargs)
            self.obs = raw_obs
            self.cumulative_reward += reward

            for i, ((a, e), ai) in enumerate(results):
                if ai.get("failed"):  # add a penalty for failing to generate a plan
                    self.cumulative_reward[i] -= 10
                if "subgoal" in ai:
                    self.action_queue.add_item(
                        (ai["rnn_hxs"][i], ai["subgoal"], ai["probs"], e, ai["value"]),
                        i,
                    )
                if (done[i] and valid_envs[i]) or "empty" in ai:
                    if done[i]:
                        self._policy.reset(i)
                    self.obs_queue.add_item(
                        (self.obs[i], self.cumulative_reward[i], done[i], infos[i]), i
                    )
                    self.cumulative_reward[i] = 0
            step_count += 1
            # print(step_count)

        # [
        #     print(f"obs queue layer {i} length {len(q)}")
        #     for i, q in enumerate(self.obs_queue.queues)
        # ]
        # [
        #     print(f"action queue layer {i} length {len(q)}")
        #     for i, q in enumerate(self.action_queue.queues)
        # ]
        o_layer = self.obs_queue.pop_layer()
        a_layer = self.action_queue.pop_layer()
        layer = [o + a for o, a in zip(o_layer, a_layer)]
        obs, reward, done, infos, recurrent_hidden_states, action, action_log_prob, explored, value = [
            z for z in zip(*layer)
        ]

        raw_obs = np.array(obs)
        recurrent_hidden_states = torch.stack(recurrent_hidden_states, dim=0)
        action = torch.cat(action)
        action_log_prob = torch.cat(action_log_prob)
        explored = torch.cat(explored)
        value = torch.cat(value)
        reward = np.array(reward)

        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
        )
        gt.stamp("exploration sampling", unique=False)

        self._rollouts.insert(
            stored_obs,
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
                self.eval_data_collector.collect_one_step(
                    step, self.num_eval_steps_per_epoch
                )
            gt.stamp("evaluation sampling")

            for _ in range(self.num_train_loops_per_epoch):
                # this if check could be moved inside the function
                if self.use_linear_lr_decay:
                    # decrease learning rate linearly
                    self.trainer.decay_lr(epoch, self.num_epochs)

                for step in range(self.num_expl_steps_per_train_loop):
                    self.expl_data_collector.collect_one_step(
                        step, self.num_expl_steps_per_train_loop
                    )
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
