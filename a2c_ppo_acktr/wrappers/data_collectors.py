import numpy as np

from gym.spaces import Tuple
import torch
import gtimer as gt

from a2c_ppo_acktr.storage import RolloutStorage

from rlkit.core.external_log import LogPathCollector
import rlkit.torch.pytorch_util as ptu
from rlkit.util.multi_queue import MultiQueue
import rlkit.pythonplusplus as ppp

from gym_taxi.utils.spaces import Json


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
        # print(action)
        # print(explored)
        # print(agent_info)

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
        gt.stamp("data storing", unique=False)
        flat_ai = ppp.dict_of_list__to__list_of_dicts(agent_info, len(action))
        gt.stamp("flattening", unique=False)
        # print(flat_ai)
        self.add_step(action, action_log_prob, reward, done, value, flat_ai)


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
        gamma=1,
        no_plan_penalty=False,
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
        self.discounts = np.ones(num_processes)
        self.plan_length = np.zeros(num_processes)
        self.gamma = gamma
        self.no_plan_penalty = no_plan_penalty

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

            action = np.array([[a] for (a, _), _ in results])
            # print(f"actions: {action}")
            # Observe reward and next obs
            raw_obs, reward, done, infos = self._env.step(action)
            if self._render:
                self._env.render(**self._render_kwargs)
            self.obs = raw_obs
            self.discounts *= self.gamma
            self.plan_length += 1
            self.cumulative_reward += reward * self.discounts
            # print("results now")

            for i, ((a, e), ai) in enumerate(results):
                # print(f"results: {i}, {((a, e), ai)}")
                if (
                    ai.get("failed") and not self.no_plan_penalty
                ):  # add a penalty for failing to generate a plan
                    self.cumulative_reward[i] -= 0.5
                if "subgoal" in ai:
                    self.action_queue.add_item(
                        (ai["rnn_hxs"], ai["subgoal"], ai["probs"], e, ai["value"], ai),
                        i,
                    )
                if (done[i] and valid_envs[i]) or "empty" in ai:
                    if done[i]:
                        self._policy.reset(i)
                    self.obs_queue.add_item(
                        (
                            self.obs[i],
                            self.cumulative_reward[i],
                            done[i],
                            infos[i],
                            self.plan_length[i],
                        ),
                        i,
                    )
                    self.cumulative_reward[i] = 0
                    self.discounts[i] = 1
                    self.plan_length[i] = 0
            step_count += 1
            # print("results done")
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
        obs, reward, done, infos, plan_length, recurrent_hidden_states, action, action_log_prob, explored, value, agent_info = [
            z for z in zip(*layer)
        ]

        raw_obs = np.array(obs)
        recurrent_hidden_states = torch.cat(recurrent_hidden_states)
        action = torch.cat(action)
        action_log_prob = torch.cat(action_log_prob)
        explored = torch.cat(explored)
        value = torch.cat(value)
        reward = np.array(reward)
        plan_length = np.array(plan_length)

        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        plan_length = torch.from_numpy(plan_length).unsqueeze(dim=1).float()

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
            plan_length,
        )
        self.add_step(action, action_log_prob, reward, done, value, agent_info)

    def get_snapshot(self):
        return dict(env=self._env, policy=self._policy.learner)

