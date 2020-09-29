#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_baselines.rl.ddppo.algo.ddppo_trainer import DDPPOTrainer
from habitat import Config, logger

from habitat_baselines.rl.ddppo_semantic.semantic_policy import ObjectNavPolicy
from habitat_baselines.rl.ddppo_semantic.ddppo_semantic import DDPPOSE



@baseline_registry.register_trainer(name="ddppo_semantic")
class DDPPOSETrainer(DDPPOTrainer):

    def __init__(self, config=None):
        super().__init__(config)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = ObjectNavPolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            pretrain_path = self.config.RL.SEDDPPO.pretrained_visual,
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=self.config.RL.DDPPO.rnn_type,
            num_recurrent_layers=self.config.RL.DDPPO.num_recurrent_layers,
            backbone=self.config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb"
            in self.envs.observation_spaces[0].spaces,
        )
        self.actor_critic.to(self.device)

        print("*************************** observation_space", self.envs.observation_spaces)
        print("*************************** action_space", self.envs.action_spaces[0].n)
        print("*************************** action_space n", self.envs.action_spaces)

        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )
        
        # print('.pth type:', type(pretrained_state))
        # print('.pth len:', len(pretrained_state))
        # print('--------------------------')
        # for k in pretrained_state.keys():
        #     print(k, type(pretrained_state[k]), pretrained_state[k])
            
        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPOSE(
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
