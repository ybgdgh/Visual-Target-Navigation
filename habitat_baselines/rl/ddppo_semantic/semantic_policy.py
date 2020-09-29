#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.utils import Flatten, ResizeCenterCropper
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo import Net, Policy
from habitat_baselines.rl.ddppo_semantic.resnet import resnet50
from habitat_baselines.rl.ddppo_semantic.depnet import DepNet
from habitat_baselines.rl.ddppo_semantic.semnet import SemNet

from collections import Counter

class ObjectNavPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        pretrain_path,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        resnet_baseplanes=32,
        backbone="resnet50",
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__(
            ObjectNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
                pretrain_path=pretrain_path,
            ),
            action_space.n,
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        hidden_size,
        pretrain_path,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__()

        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            observation_space = self.obs_transform.transform_observation_space(
                observation_space
            )

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        if "semantic" in observation_space.spaces:
            self._n_input_semantic = 1
        else:
            self._n_input_semantic = 0

        self.num_classes = 21

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(
                self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)
        self.fc_rgb = nn.Linear(2048, 512)
        self.fc = nn.Linear(1536, hidden_size)

        if not self.is_blind:
            # input_channels = self._n_input_depth + self._n_input_rgb
            self.depth_encoder = DepNet(512)
            self.semantic_encoder = SemNet(self.num_classes, 512)
            self.rgb_encoder = resnet50(pretrain_path, pretrained=True) # 1024
            for param in self.rgb_encoder.parameters(): 
                param.requires_grad = False

            # print("******************* observation_space: ", observation_space.spaces["rgb"].shape) #(256,256,3)
            # print("******************* input_channels: ", input_channels) # 4

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)
            

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        if self._n_input_semantic > 0:
            semantic_observations = observations["semantic"]
            semantic_observations = semantic_observations.reshape(
                semantic_observations.shape[0], 
                semantic_observations.shape[1], 
                semantic_observations.shape[2], 
                1
                )
            # print("semantic_observations: ", semantic_observations.shape)
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            semantic_observations = semantic_observations.permute(0, 3, 1, 2)

            # print("semantic_observations type: {}".format(type(semantic_observations))) # <class 'torch.Tensor'>

            # detach the semantic map to the number of task id
            se_o = []
            for i in range(self.num_classes):
                se = semantic_observations.clone()
                # print(se[se!=0])
                se[se!=(i+1)]=0
                se[se==(i+1)]=1
                # se = se.view(semantic_observations.size(0), semantic_observations.size(2), semantic_observations.size(3))
                se_o.append(se.clone())
                # print(se[se!=0]) # 查看数据是否分层

            # print("******se_o = ", se_o[0].shape) #torch.Size([4, 1, 256, 256])

            se_o_input = torch.cat(se_o, dim=1) 
            # print("******se_o = ", se_o_input.shape) #torch.Size([4, 21, 256, 256])
            # cnn_input.append(semantic_observations)
            cnn_input.append(se_o_input)

        if self.obs_transform:
            cnn_input = [self.obs_transform(inp) for inp in cnn_input]

        # x = torch.cat(cnn_input, dim=1)
        # print("***********x: ", x.shape)
        cnn_input[0] = F.avg_pool2d(cnn_input[0], 2) # 4 3 128 128 屏蔽后显存溢出
        cnn_input[1] = F.avg_pool2d(cnn_input[1], 2) # 4 1 128 128 屏蔽后显存溢出
        cnn_input[2] = F.max_pool2d(cnn_input[2], 2) # 4 21 128 128 屏蔽后显存溢出

        cnn_input[0] = self.running_mean_and_var(cnn_input[0]) # 4 3 128 128
        # cnn_input[2] = cnn_input[2]/100.0

        # print("*********** cnn_input :", len(cnn_input))  # 2
        # print("*********** cnn_input 0:", cnn_input[0].shape) # torch.Size([4, 3, 128, 128])
        # print("*********** cnn_input 1:", cnn_input[1].shape) # torch.Size([4, 1, 128, 128])
        # print("*********** cnn_input 2:", cnn_input[2].shape) # torch.Size([4, 1, 128, 128])


        cnn_input[0] = self.rgb_encoder(cnn_input[0]) # 2048
        cnn_input[0] = self.fc_rgb(cnn_input[0]) # 512
        cnn_input[1] = self.depth_encoder(cnn_input[1]) # 512
        cnn_input[2] = self.semantic_encoder(cnn_input[2]) # 512

        # print("rgb: ", cnn_input[0].shape) # 4 512
        # print("depth: ", cnn_input[1].shape) # 4 512

        # x = cnn_input[0]
        x = torch.cat(cnn_input, dim=1) 
        x = self.relu(x)
        x = self.fc(x) # 2048 -> 512
        
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = torch.where(torch.isinf(x), torch.full_like(x, 0), x)
        return x


class ObjectNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
        pretrain_path,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__()

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        self._hidden_size = hidden_size

        self.visual_encoder = ResNetEncoder(
            observation_space,
            hidden_size,
            pretrain_path,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            normalize_visual_inputs=normalize_visual_inputs,
            obs_transform=obs_transform,
        )
        rnn_input_size += hidden_size
        # print("rnn_input_size: ", rnn_input_size) #640 = 512 + 32 * 4
        self.state_encoder = RNNStateEncoder(
            rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        # print("*************observations: ", observations) # 4 512
        # print("*************observations: ", len(observations['rgb'])) # 4 512

        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            # print("*************visual_feats: ", visual_feats.shape) # 4 512

            x.append(visual_feats)

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            # print("***************object_goal: ", self.obj_categories_embedding(object_goal).squeeze(dim=1).shape) #4*32
            # print("***************object_goal: ", object_goal) # 4*1
            # print("***************object_goal: ", object_goal.shape) # 4*1
            # print("***************_n_object_categories: ", self._n_object_categories) # 21
            # object_goal = object_goal-1
            # one_hot_goal = torch.zeros(object_goal.shape[0], self._n_object_categories).scatter_(1, object_goal.cpu(), 1)
            # one_hot_goal = torch.nn.functional.one_hot(object_goal, self._n_object_categories).squeeze(dim=1).float()
            # print("***************one hot: ", one_hot_goal)
            # print("***************one hot: ", one_hot_goal.shape) # 4*21
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
        )
        x.append(prev_actions)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


