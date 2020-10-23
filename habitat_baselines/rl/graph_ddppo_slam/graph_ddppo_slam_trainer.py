import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict


from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry


from habitat_baselines.rl.ddppo_slam.ddppo_slam_trainer import DDPPOSLAMTrainer
from habitat_baselines.rl.ddppo_slam.ddppo_slam import DDPPOSLAM
from habitat_baselines.rl.graph_ddppo_slam.graph_slam_policy import ObjectNavGraphSLAMPolicy

@baseline_registry.register_trainer(name="graph_ddppo_slam")
class GRAPHDDPPOSLAMTrainer(DDPPOSLAMTrainer):
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
        # print("*************************** self.obs_space:", self.obs_space) #self.obs_space: Dict(compass:Box(1,), depth:Box(256, 256, 1), gps:Box(2,), map_sum:Box(480, 480, 23), objectgoal:Box(1,), rgb:Box(256, 256, 3), semantic:Box(256, 256))


        # Global policy action space
        self.g_action_space = spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)
    
        self.actor_critic = ObjectNavGraphSLAMPolicy(
            observation_space=self.obs_space,
            g_action_space=self.g_action_space,
            l_action_space=self.envs.action_spaces[0],
            pretrain_path = None,
            output_size = self.config.RL.SLAMDDPPO.map_output_size,
        )
        self.actor_critic.to(self.device)

        print("*************************** action_space", self.envs.action_spaces[0].n)
        # print("*************************** action_space n", self.envs.action_spaces)

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
  
        # print("*************************** num_steps:", ppo_cfg.num_steps)
        # print("*************************** num_envs:", self.envs.num_envs)
        # print("*************************** self.obs_space:", self.obs_space.spaces)
        # print("*************************** action_spaces:", self.envs.action_spaces[0])
        # print("*************************** hidden_size:", ppo_cfg.hidden_size)

