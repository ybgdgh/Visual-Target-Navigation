#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch.optim.lr_scheduler import LambdaLR

import tqdm
import contextlib
import os
import random
import time
from collections import OrderedDict, defaultdict, deque

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.algo.ddppo_trainer import DDPPOTrainer
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.global_rollout_storage import GlobalRolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)

from habitat_baselines.rl.ddppo_slam.slam_policy import ObjectNavSLAMPolicy
from habitat_baselines.rl.ddppo_slam.ddppo_slam import DDPPOSLAM




@baseline_registry.register_trainer(name="ddppo_slam")
class DDPPOSLAMTrainer(DDPPOTrainer):

    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None):
        super().__init__(config)

    def _setup_actor_critic_agent(self, observations, ppo_cfg: Config) -> None:
        # Global policy observation space
        self.obs_space = self.envs.observation_spaces[0]
        # add the map observation space
        self.obs_space = SpaceDict(
            {
                "map_sum": spaces.Box(
                    low=0,
                    high=1,
                    shape=observations[0]["map_sum"].shape,
                    dtype=np.uint8,
                ),
                "curr_pose": spaces.Box(
                    low=0,
                    high=1,
                    shape=observations[0]["curr_pose"].shape,
                    dtype=np.uint8,
                ),
                **self.obs_space.spaces,
            }
        )
        print("*************************** self.obs_space:", self.obs_space) #self.obs_space: Dict(compass:Box(1,), depth:Box(256, 256, 1), gps:Box(2,), map_sum:Box(480, 480, 23), objectgoal:Box(1,), rgb:Box(256, 256, 3), semantic:Box(256, 256))


        # Global policy action space
        self.g_action_space = spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)
    
        self.actor_critic = ObjectNavSLAMPolicy(
            observation_space=self.obs_space,
            g_action_space=self.g_action_space,
            l_action_space=self.envs.action_spaces[0],
            pretrain_path = None,
            output_size = self.config.RL.SLAMDDPPO.map_output_size,
        )
        self.actor_critic.to(self.device)

        print("*************************** action_space", self.envs.action_spaces[0].n)
        # print("*************************** action_space n", self.envs.action_spaces)


        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPOSLAM(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )
  
        print("*************************** num_steps:", ppo_cfg.num_steps)
        print("*************************** num_envs:", self.envs.num_envs)
        print("*************************** self.obs_space:", self.obs_space.spaces)
        print("*************************** action_spaces:", self.envs.action_spaces[0])
        print("*************************** hidden_size:", ppo_cfg.hidden_size)


    
    def _collect_global_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        # Run Global Policy (global_goals = Long-Term Goal) 16
        if rollouts.step % (self.num_each_global_step) == 0:
            # print("rollouts.step: ", rollouts.step)
            with torch.no_grad():
                step_observation = {
                    k: v[rollouts.step] for k, v in rollouts.observations.items()
                }

                (
                    self.values,
                    self.actions,
                    self.actions_log_probs,
                ) = self.actor_critic.act(
                    step_observation,
                    rollouts.prev_g_actions[rollouts.step],
                    rollouts.masks[rollouts.step],
                    # deterministic=True,
                )
            
            self.envs.update_full_map()

            self.global_goals = torch.Tensor(
                        [[(action[0] * self.map_w), 
                        (action[1] * self.map_h)]
                        for action in self.actions])
        # print("global_goals: ", self.global_goals)
            # print("self.actions: ", self.actions)
            # print("actions_log_probs: ", self.actions_log_probs)

        # sample action
        l_actions = self.envs.get_local_actions(self.global_goals)

        # print("action: ", l_actions)

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step(l_actions)

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        # print("step:", rollouts.step,
        #     "l_actions:", l_actions[0], 
        #     "global_goals:", self.global_goals[0],
        #     "pose:", observations[0]["curr_pose"], 
        #     "object:", observations[0]["objectgoal"],
        #     "dones:", dones[0])

        # print("dones: ", dones)
        env_time += time.time() - t_step_env

        t_update_stats = time.time()
  
        batch = batch_obs(observations, device=self.device)

        # assert (np.any(np.isnan(rewards)) == False), "Reward has nan."

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        rewards = torch.where(torch.isnan(rewards), torch.full_like(rewards, 0), rewards)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks
        
        rollouts.insert(
            batch,
            self.actions,
            self.actions_log_probs,
            self.values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs



    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.prev_g_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()
                    
        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        # print("next_value: ", next_value)

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()
        # print("update rollouts")
        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )


    def train(self) -> None:
        r"""Main method for DD-PPO SLAM.

        Returns:
            None
        """

        #####################################################################
        ## init distrib and configuration #####################################################################
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        # self.local_rank = 1
        add_signal_handlers()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank() # server number
        self.world_size = distrib.get_world_size() 

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank # gpu number in one server
        self.config.SIMULATOR_GPU_ID = self.local_rank
        print("********************* TORCH_GPU_ID: ", self.config.TORCH_GPU_ID)
        print("********************* SIMULATOR_GPU_ID: ", self.config.SIMULATOR_GPU_ID)

        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_PROCESSES
        )
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")


        #####################################################################
        ## build distrib NavSLAMRLEnv environment
        #####################################################################
        print("#############################################################")
        print("## build distrib NavSLAMRLEnv environment")
        print("#############################################################")
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        observations = self.envs.reset()
        print("*************************** observations len:", len(observations))

        # semantic process
        for i in range(len(observations)):
            observations[i]["semantic"] = observations[i]["semantic"].astype(np.int32)
            se = list(set(observations[i]["semantic"].ravel()))
            print(se)
        # print("*************************** observations type:", observations)
        # print("*************************** observations type:", observations[0]["map_sum"].shape) # 480*480*23
        # print("*************************** observations curr_pose:", observations[0]["curr_pose"]) # []

        batch = batch_obs(observations, device=self.device)
        print("*************************** batch len:", len(batch))
        # print("*************************** batch:", batch)

        # print("************************************* current_episodes:", (self.envs.current_episodes()))

        #####################################################################
        ## init actor_critic agent
        #####################################################################  
        print("#############################################################")
        print("## init actor_critic agent")
        print("#############################################################")
        self.map_w = observations[0]["map_sum"].shape[0]
        self.map_h = observations[0]["map_sum"].shape[1]
        # print("map_: ", observations[0]["curr_pose"].shape)


        ppo_cfg = self.config.RL.PPO
        if (
            not os.path.isdir(self.config.CHECKPOINT_FOLDER)
            and self.world_rank == 0
        ):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(observations, ppo_cfg)

        self.agent.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )

        #####################################################################
        ## init Global Rollout Storage
        #####################################################################  
        print("#############################################################")
        print("## init Global Rollout Storage")
        print("#############################################################") 
        self.num_each_global_step = self.config.RL.SLAMDDPPO.num_each_global_step
        rollouts = GlobalRolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.g_action_space,
        )
        rollouts.to(self.device)

        print('rollouts type:', type(rollouts))
        print('--------------------------')
        # for k in rollouts.keys():
        # print("rollouts: {0}".format(rollouts.observations))

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
    
            _, actions, _, = self.actor_critic.act(
                step_observation,
                rollouts.prev_g_actions[0],
                rollouts.masks[0],
            )

        self.global_goals = [[int(action[0].item() * self.map_w), 
                            int(action[1].item() * self.map_h)]
                            for action in actions]

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        print("*************************** current_episode_reward:", current_episode_reward)
        print("*************************** running_episode_stats:", running_episode_stats)
        # print("*************************** window_episode_stats:", window_episode_stats)


        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        # interrupted_state = load_interrupted_state("/home/cirlab1/userdir/ybg/projects/habitat-api/data/interrup.pth")
        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        deif = {}
        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )
                # print("************************************* current_episodes:", type(self.envs.count_episodes()))
                
                # print(EXIT.is_set())
                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            ),
                            "/home/cirlab1/userdir/ybg/projects/habitat-api/data/interrup.pth"
                        )
                    print("********************EXIT*********************")

                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_global_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    # print("************************************* current_episodes:")

                    for i in range(len(self.envs.current_episodes())):
                        # print(" ", self.envs.current_episodes()[i].episode_id," ", self.envs.current_episodes()[i].scene_id," ", self.envs.current_episodes()[i].object_category)
                        if self.envs.current_episodes()[i].scene_id not in deif:
                            deif[self.envs.current_episodes()[i].scene_id]=[int(self.envs.current_episodes()[i].episode_id)]
                        else:
                            deif[self.envs.current_episodes()[i].scene_id].append(int(self.envs.current_episodes()[i].episode_id))


                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break

                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()
                if self._static_encoder:
                    self._encoder.eval()

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                stats_ordering = list(sorted(running_episode_stats.keys()))
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )
                distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [value_loss, action_loss, count_steps_delta],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += stats[2].item()

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    losses = [
                        stats[0].item() / self.world_size,
                        stats[1].item() / self.world_size,
                    ]
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    writer.add_scalar(
                        "reward",
                        deltas["reward"] / deltas["count"],
                        count_steps,
                    )

                    # Check to see if there are any metrics
                    # that haven't been logged yet
                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in {"reward", "count"}
                    }
                    if len(metrics) > 0:
                        writer.add_scalars("metrics", metrics, count_steps)

                    writer.add_scalars(
                        "losses",
                        {k: l for l, k in zip(losses, ["value", "policy"])},
                        count_steps,
                    )

                    # log stats
                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                update,
                                count_steps
                                / ((time.time() - t_start) + prev_time),
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(
                                update, env_time, pth_time, count_steps
                            )
                        )
                        logger.info(
                            "Average window size: {}  {}".format(
                                len(window_episode_stats["count"]),
                                "  ".join(
                                    "{}: {:.3f}".format(k, v / deltas["count"])
                                    for k, v in deltas.items()
                                    if k != "count"
                                ),
                            )
                        )

                        # for k in deif:
                        #     deif[k] = list(set(deif[k]))
                        #     deif[k].sort()
                        #     print("deif: k", k, " : ", deif[k])

                    # checkpoint model
                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth",
                            dict(step=count_steps),
                        )
                        print('=' * 20 + 'Save Model' + '=' * 20)
                        logger.info(
                            "Save Model : {}".format(count_checkpoints)
                        )
                        count_checkpoints += 1

            self.envs.close()



    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        print("Checkpoint: {}".format(ckpt_dict))

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        self.map_w = observations[0]["map_sum"].shape[0]
        self.map_h = observations[0]["map_sum"].shape[1]

        self._setup_actor_critic_agent(observations, ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            512,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 2, device=self.device, dtype=torch.long
        )

        global_goals = [[int(action[0].item() * self.map_w), 
                            int(action[1].item() * self.map_h)]
                            for action in prev_actions]

        
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        self.step=0
        self.num_each_global_step = self.config.RL.SLAMDDPPO.num_each_global_step
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            if self.step % (self.num_each_global_step) == 0:
                with torch.no_grad():
                    (
                        _,
                        actions,
                        _,
                    ) = self.actor_critic.act(
                        batch,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )

                    prev_actions.copy_(actions)
                self.step=0
                self.envs.update_full_map()

                global_goals = torch.Tensor(
                        [[(action[0] * self.map_w), 
                        (action[1] * self.map_h)]
                        for action in actions])

            l_actions = self.envs.get_local_actions(global_goals)
            self.step = self.step+1

            outputs = self.envs.step(l_actions)
            

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    print("episode_stats", episode_stats)
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    
                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            object_category=current_episodes[i].object_category,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        for k, v in stats_episodes.items():
            print(k, ": ", v)
        # print("stats_episodes: ", stats_episodes)
        print("num_episodes: {}".format(num_episodes))
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )
            # print("sum: {}".format(sum([v[stat_key] for v in stats_episodes.values()])))

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()
