import numpy as np

from gym.spaces import Tuple, Box
import torch
import gtimer as gt

from a2c_ppo_acktr.storage import RolloutStorage, AsyncRollouts

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

        self.shape = (
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
            self.shape,
            env.action_space,
            1,  # hardcoding reccurent hidden state off for now.
        )

        raw_obs = env.reset()
        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs
        self.obs = (
            raw_obs
            if isinstance(self, HierarchicalStepCollector)
            or isinstance(self, ThreeTierStepCollector)
            else action_obs
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


class ThreeTierStepCollector(RolloutStepCollector):
    def __init__(
        self,
        env,
        policy,
        device,
        ancillary_goal_size,
        symbolic_action_size,
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

        self._learn_rollouts = AsyncRollouts(
            max_num_epoch_paths_saved,
            num_processes,
            self.shape,
            Box(-np.inf, np.inf, (ancillary_goal_size,)),
            1,  # hardcoding reccurent hidden state off for now.
        )

        self._learn_rollouts.obs[0].copy_(self._rollouts.obs[0])
        self._learn_rollouts.to(self.device)

        if env.action_space.__class__.__name__ == "Discrete":
            self.action_size = 1
        else:
            self.action_size = env.action_space.shape[0]

        self.ancillary_goal_size = ancillary_goal_size
        n, p, l = self._rollouts.obs.shape
        self._rollouts.obs = torch.zeros(n, p, l + symbolic_action_size).to(self.device)
        self.symbolic_actions = torch.zeros(p, symbolic_action_size).to(self.device)

    def get_rollouts(self):
        return self._rollouts, self._learn_rollouts

    def parallelise_results(self, results):
        action = torch.zeros((self.num_processes, self.action_size))
        explored = torch.zeros((self.num_processes, 1))
        value = torch.zeros((self.num_processes, 1))
        action_log_prob = torch.zeros((self.num_processes, 1))
        recurrent_hidden_states = torch.zeros((self.num_processes, 1))

        invalid = torch.zeros((self.num_processes, 1))
        goal = torch.zeros((self.num_processes, self.ancillary_goal_size))
        learn_value = torch.zeros((self.num_processes, 1))
        learn_action_log_prob = torch.zeros((self.num_processes, 1))
        learn_recurrent_hidden_states = torch.zeros((self.num_processes, 1))
        plan_mask = np.zeros((self.num_processes,), dtype=np.bool)

        for i, ((a, e), aic) in enumerate(results):
            ail = aic["agent_info_learn"]
            action[i] = a
            explored[i] = e
            value[i] = aic["value"]
            action_log_prob[i] = aic["probs"]
            print("buffer", recurrent_hidden_states.shape)
            print("rnnhxs", aic["rnn_hxs"])
            print("probs", aic["probs"])
            recurrent_hidden_states = aic["rnn_hxs"]
            self.symbolic_actions[i] = aic["symbolic_action"]
            if "subgoal" in ail:
                plan_mask[i] = 1
                invalid[i] = ail["failed"]
                goal[i] = ail["subgoal"]
                learn_value[i] = ail["value"]
                learn_action_log_prob[i] = ail["probs"]
                learn_recurrent_hidden_states = ail["rnn_hxs"]

        return (
            action,
            explored,
            value,
            action_log_prob,
            recurrent_hidden_states,
            invalid,
            goal,
            learn_value,
            learn_action_log_prob,
            learn_recurrent_hidden_states,
            plan_mask,
        )

    def collect_one_step(self, step, step_total):
        """
        This needs to be aware of both high and low level experience, as well as internal and external rewards. 
        High level experience is: S,G,S',R
        Low level experience is: s,a,s',r
        every frame always goes into low level experience, and rewards are calculated as external + completion bonus.

        Will always be a single pass, as low level actions are always generated. What about when there is no valid plan?

        What does the interface between this and learn_plan_policy need to be?
            - action selected
            - goal selected (if any) - if the goal is selected, then add last frames observation to the buffer. 
            - invalid goal selected - if so, penalise learner (potentially)
            - symbolic-step timeout (true/false) - add negative reward to learner
            - symbolic-step completed (true/false) - add positive reward to controller
            - agent info / explored for both high and low levels. 
        """

        with torch.no_grad():
            results = self._policy.get_action(
                self.obs, valid_envs=[True] * self.num_processes
            )

        (
            action,
            explored,
            value,
            action_log_prob,
            recurrent_hidden_states,
            invalid,
            goal,
            learn_value,
            learn_action_log_prob,
            learn_recurrent_hidden_states,
            plan_mask,
        ) = self.parallelise_results(results)

        # value = agent_info_control["value"]
        # action_log_prob = agent_info_control["probs"]
        # recurrent_hidden_states = agent_info_control["rnn_hxs"]

        # agent_info_learn = agent_info_control.pop("agent_info_learn")

        # if there was any planning that step:

        if any(plan_mask):
            # handle invalid plans if desired
            self._learn_rollouts.action_insert(
                learn_recurrent_hidden_states,
                goal,
                learn_action_log_prob,
                learn_value,
                plan_mask,
            )

        # Observe reward and next obs
        raw_obs, reward, done, infos = self._env.step(ptu.get_numpy(action))
        if self._render:
            self._env.render(**self._render_kwargs)

        # perform observation conversions
        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs
        augmented_obs = torch.cat((stored_obs, self.symbolic_actions), axis=1)
        self.obs = raw_obs

        # check to see whether symbolic actions are complete
        step_timeout, step_complete, plan_ended = self._policy.check_action_status(
            self.obs.squeeze(1)
        )
        print("plan ended", plan_ended)
        print("reward shape", self.cumulative_reward.shape)

        internal_reward = reward + np.array(step_complete) * 10
        external_reward = reward - np.array(step_timeout) * 10

        self.discounts *= self.gamma
        self.plan_length += 1
        self.cumulative_reward += external_reward * self.discounts

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
        )

        gt.stamp("exploration sampling", unique=False)

        internal_reward = torch.from_numpy(internal_reward).unsqueeze(dim=1).float()

        self._rollouts.insert(
            augmented_obs,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            internal_reward,
            masks,
            bad_masks,
        )

        if np.any(plan_ended):
            self._learn_rollouts.observation_insert(
                stored_obs,
                torch.from_numpy(self.cumulative_reward).unsqueeze(dim=1).float(),
                masks,
                bad_masks,
                plan_ended,
                plan_length=torch.from_numpy(self.plan_length).unsqueeze(dim=1).float(),
            )

        # TODO: this is doing logging of stats for tb.... will figure it out later
        gt.stamp("data storing", unique=False)
        # flat_ai = ppp.dict_of_list__to__list_of_dicts(agent_info_control, len(action))

        # self.add_step(action, action_log_prob, external_reward, done, value, flat_ai)

        # reset plans
        self.cumulative_reward[plan_ended] = 0
        self.discounts[plan_ended] = 1
        self.plan_length[plan_ended] = 0

    def get_snapshot(self):
        return dict(env=self._env, policy=self._policy.learner)
