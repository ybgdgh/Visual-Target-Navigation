import abc

import torch
import torch.nn as nn

from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.utils import CategoricalNet, Flatten, DiagGaussian
from habitat_baselines.common.utils import Flatten, ResizeCenterCropper


class ObjectNavSLAMPolicy(nn.Module):
    def __init__(
        self, 
        observation_space,
        g_action_space,
        l_action_space,
        pretrain_path,
        output_size=512,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__()
    
        self.net = ObjectNavSLAMNet(
                observation_space=observation_space,
                g_action_space=g_action_space,
                output_size=output_size,
                obs_transform=obs_transform,
                pretrain_path=pretrain_path,
            )

        self.num_local_actions = g_action_space.shape[0]
        self.num_global_actions = g_action_space.shape[0]
        # print("num_global_actions: %d" % self.num_global_actions) #2

        self.global_action_distribution = DiagGaussian(
            self.net.output_size, self.num_global_actions
        )
        self.critic = CriticHead(self.net.output_size)



    def act(
        self,
        observations,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features = self.net(
            observations, prev_actions, masks
        )
        distribution = self.global_action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        # map_sum = observations["map_sum"]
        # object_ind = observations["objectgoal"]
        # # current_pose = observations["curr_pose"]
        # # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        # map_sum = map_sum.permute(0, 3, 1, 2)
        # print("observations: ", map_sum.shape) #torch.Size([2, 23, 480, 480])
        # print("object_ind: ", object_ind)
        # for index in range(len(object_ind)):
        #     # print("map_size: ", map_sum[index, 1].shape) # 480*480
        #     # print("map_objectid: ", object_ind[index][0])
        #     object_map = map_sum[index, int(object_ind[index][0])]
        #     if len(object_map[object_map!=0]) > 0:
        #         print("map_objectid: ", object_ind[index][0],
        #             "num: ", len(object_map[object_map!=0]))
        #         # print("action: ", torch.nonzero(object_map).sum(0))
        #         action[index] = (torch.nonzero(object_map).sum(0)).float() / (len(object_map[object_map!=0]) * 480.0)
        # print("action: ", action)

        action_log_probs = distribution.log_probs(action)

        # print("action_log_probs: ", action_log_probs)

        return value, action, action_log_probs


    def get_value(self, observations, prev_actions, masks):
        features = self.net(
            observations, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, prev_actions, masks, action
    ):
        features = self.net(
            observations, prev_actions, masks
        )

        # print("features: ", torch.max(features))

        distribution = self.global_action_distribution(features)
        value = self.critic(features)
        # print("evaluate_actions: ", action)
        action_log_probs = distribution.log_probs(action)
        # print("evaluate_actions_log_probs: ", action_log_probs)

        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy

class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

class MapEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super(MapEncoder, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.fc = nn.Linear(7200, out_channels)
        # self.fc = nn.Linear(1568, out_channels)
        
    def forward(self, x):
        # print("x: ", x.shape) # 1 23 480 480
        # x = self.maxpool(x) # 1 23 240 240
        x = self.conv1(x)
        x = nn.ReLU()(x)
        # print("x: ", x.shape) # 1 32 240 240

        x = self.maxpool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        # print("x: ", x.shape) # 1 64 120 120

        x = self.maxpool(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        # print("x: ", x.shape) # 1 128 60 60

        x = self.maxpool(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        # print("x: ", x.shape) # 1 64 30 30

        x = self.maxpool(x)
        x = self.conv5(x)
        x = nn.ReLU()(x)
        # print("x: ", x.shape) # 1 32 15 15
        # print("x: ", x.shape) # 1 32 7 7

        x = Flatten()(x.contiguous())

        # print("x: ", x.shape) # 1*7200

        x = self.fc(x) # 1*512
        x = nn.ReLU()(x)

        return x


class ObjectNavSLAMNet(Net):
    def __init__(
        self,
        observation_space,
        g_action_space,
        output_size,
        pretrain_path,
        obs_transform=ResizeCenterCropper(size=(240, 240)),
    ):
        super().__init__()

        self._output_size = output_size

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 2
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 256
            )
            hidden_size = 256

        # current pose embedding
        curr_pose_dim = observation_space.spaces["curr_pose"].shape[0]
        # print("curr_pose_dim: ", curr_pose_dim)
        self.curr_pose_embedding = nn.Linear(curr_pose_dim, 256)
        hidden_size += 256

        map_dim = observation_space.spaces["map_sum"].shape[2]
        self.map_encoder = MapEncoder(
            map_dim,
            2048,
            obs_transform=obs_transform,
            )

        hidden_size += 2048

        self.linear1 = nn.Linear(hidden_size, 1024)
        self.linear2 = nn.Linear(1024, self._output_size)


    @property
    def output_size(self):
        return self._output_size

    def forward(self, observations, prev_actions, masks):
        x = []

        map_sum = observations["map_sum"]

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        map_sum = map_sum.permute(0, 3, 1, 2)
        # print("map_sum shape: ", map_sum) # 1*480*480*3
        # print("self.map_embedding(map_sum) shape: ", self.map_encoder(map_sum)) # 1*480*480*3

        map_aa = self.map_encoder(map_sum)
        # print("self.map_encoder(map_sum): ", torch.max(map_aa))

        x.append(map_aa)
        
        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            # print("object_goal: ", object_goal)
            goal_aa = self.obj_categories_embedding(object_goal).squeeze(dim=1)
            # print("***************object_goal: ", goal_aa)
            x.append(goal_aa)

        # current pose embedding
        curr_pose = observations["curr_pose"]
        curr_pose = curr_pose/map_sum.shape[2]
        # print("curr_pose: ", curr_pose)
        curr_pose_obs = self.curr_pose_embedding(curr_pose)
        # print("curr_pose shape: ", curr_pose_obs) # 1*2
        # print("curr_pose: ", torch.max(curr_pose_obs)) # 1*2
        # print("curr_pose_obs shape: ", curr_pose_obs.shape) # 1*32
        
        x.append(curr_pose_obs)
        # print("x: ", x)

        x = torch.cat(x, dim=1)

        x = nn.ReLU()(self.linear1(x))

        x = nn.ReLU()(self.linear2(x))

        return x
