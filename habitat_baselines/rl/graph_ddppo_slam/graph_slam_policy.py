import abc
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.data import Data

from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.utils import CategoricalNet, Flatten, DiagGaussian

from habitat_baselines.common.utils import Flatten, ResizeCenterCropper
from habitat_baselines.rl.ddppo_slam.slam_policy import ObjectNavSLAMPolicy, MapEncoder, Net
from habitat_baselines.rl.graph_ddppo_slam.graphcnn import GraphCNN, GraphRCNN


task_category = [
    "chair",
    "plant",
    "sink",
    "vase",
    "book",
    "couch",
    "bed",
    "bottle",
    "table",
    "toilet",
    "refrigerator",
    "tv",
    "clock",
    "oven",
    "bowl",
    "cup",
    "bench",
    "microwave",
    "suitcase",
    "umbrella",
    "teddy bear"
]

class ObjectNavGraphSLAMPolicy(ObjectNavSLAMPolicy):
    def __init__(
        self, 
        observation_space,
        g_action_space,
        l_action_space,
        pretrain_path,
        output_size=512,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__(observation_space,
                g_action_space,
                l_action_space,
                pretrain_path,
                output_size=512,
                obs_transform=ResizeCenterCropper(size=(256, 256)),
                )

        self.net = ObjectNavGraphSLAMNet(
                observation_space=observation_space,
                g_action_space=g_action_space,
                output_size=output_size,
                obs_transform=obs_transform,
                pretrain_path=pretrain_path,
            )


class ObjectNavGraphSLAMNet(Net):
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

        # goal encoder
        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            # self.fasttext = torchtext.vocab.FastText()
            # self.obj_categories_encoder = nn.Linear(
            #     300, 512
            # )
            # hidden_size = 512
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
        # print('obj_categories_embedding: ', self.obj_categories_embedding) # 22

        curr_pose_dim = observation_space.spaces["curr_pose"].shape[0]
        # print("curr_pose_dim: ", curr_pose_dim)
        self.curr_pose_embedding = nn.Linear(curr_pose_dim, 256)
        hidden_size += 256

        # map encoder
        map_dim = observation_space.spaces["map_sum"].shape[2]
        self.map_encoder = MapEncoder(
            map_dim,
            2048,
            obs_transform=obs_transform,
        )

        hidden_size += 2048

        # scene priors
        self.semantic_encoder = SemanticMap_Encoder(1, 256)
        self.graphcnn = GraphRCNN(512, 21)
        self.edge = torch.tensor([
            [0, 8, 0, 19, 0, 11, 0, 6, 0, 1, 0, 5, 0, 16, 0, 3, 0, 10, 0, 15, 0, 7, 0, 9, 0, 20, 0, 12, 5, 20, 6, 20, 8, 20, 4, 20, 16, 20, 7, 20, 15, 20, 18, 20, 8, 14, 7, 14, 1, 14, 2, 14, 9, 14, 6, 14, 3, 14, 14, 17, 5, 14, 13, 14, 10, 14, 14, 15, 4, 14, 7, 8, 4, 7, 2, 7, 7, 10, 7, 9, 1, 7, 6, 7, 7, 16, 7, 15, 7, 17, 5, 7, 7, 13, 7, 18, 3, 8, 1, 3, 3, 6, 2, 3, 3, 10, 3, 15, 3, 4, 4, 8, 0, 4, 4, 6, 4, 11, 4, 18, 4, 16, 4, 12, 4, 5, 2, 8, 2, 9, 2, 17, 1, 2, 2, 10, 5, 8, 8, 11, 8, 15, 8, 19, 6, 8, 1, 8, 8, 16, 8, 9, 8, 12, 8, 13, 8, 18, 8, 10, 2, 15, 15, 16, 1, 15, 9, 15, 15, 17, 12, 15, 13, 15, 1, 5, 1, 16, 1, 11, 1, 10, 1, 9, 16, 19, 18, 19, 3, 19, 6, 11, 6, 18, 5, 6, 6, 16, 13, 17, 10, 17, 12, 17, 8, 17, 10, 11, 5, 11, 10, 13, 12, 13, 6, 12, 11, 12, 12, 16, 16, 18, 0, 18],
            [8, 0, 19, 0, 11, 0, 6, 0, 1, 0, 5, 0, 16, 0, 3, 0, 10, 0, 15, 0, 7, 0, 9, 0, 20, 0, 12, 0, 20, 5, 20, 6, 20, 8, 20, 4, 20, 16, 20, 7, 20, 15, 20, 18, 14, 8, 14, 7, 14, 1, 14, 2, 14, 9, 14, 6, 14, 3, 17, 14, 14, 5, 14, 13, 14, 10, 15, 14, 14, 4, 8, 7, 7, 4, 7, 2, 10, 7, 9, 7, 7, 1, 7, 6, 16, 7, 15, 7, 17, 7, 7, 5, 13, 7, 18, 7, 8, 3, 3, 1, 6, 3, 3, 2, 10, 3, 15, 3, 4, 3, 8, 4, 4, 0, 6, 4, 11, 4, 18, 4, 16, 4, 12, 4, 5, 4, 8, 2, 9, 2, 17, 2, 2, 1, 10, 2, 8, 5, 11, 8, 15, 8, 19, 8, 8, 6, 8, 1, 16, 8, 9, 8, 12, 8, 13, 8, 18, 8, 10, 8, 15, 2, 16, 15, 15, 1, 15, 9, 17, 15, 15, 12, 15, 13, 5, 1, 16, 1, 11, 1, 10, 1, 9, 1, 19, 16, 19, 18, 19, 3, 11, 6, 18, 6, 6, 5, 16, 6, 17, 13, 17, 10, 17, 12, 17, 8, 11, 10, 11, 5, 13, 10, 13, 12, 12, 6, 12, 11, 16, 12, 18, 16, 18, 0]
            ], dtype=torch.long)

        self.edge_type = torch.tensor(
            [2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            dtype=torch.long)

        self.graph_fc = nn.Linear(21*21, 256)

        hidden_size += 256

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
        # print("map_sum shape: ", map_sum.shape) # 1*480*480*3
        # /print("self.map_embedding(map_sum) shape: ", self.map_encoder(map_sum).shape) # 1*480*480*3

        x.append(self.map_encoder(map_sum))
        
        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            # print("Object_goal: ", object_goal.device) #<class 'torch.Tensor'>
            # word_vecs = torch.stack([self.fasttext.vectors[self.fasttext.stoi[task_category[x-1]]] for x in object_goal[:,0]])
            # word_vecs = word_vecs.to(object_goal.device)
            # print(word_vecs.shape)
            # print("***************object_goal: ", self.obj_categories_encoder(word_vecs).shape)
            # x.append(self.obj_categories_encoder(word_vecs))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))


        # semantic map encoder
        semantic_map_sum = observations["map_sum"][:, :, :, 1:22]
        semantic_map_sum = semantic_map_sum.permute(0, 3, 1, 2)
        # print("semantic_map_sum: ", semantic_map_sum.shape) # 1 21 240 240
        semantic_map_encoder = torch.stack(
                                [self.semantic_encoder(semantic_map_sum[i].unsqueeze(1))
                                for i in range(semantic_map_sum.shape[0])]
                                )

        # print("semantic_map_encoder: " , semantic_map_encoder.shape) # 1 21 256

        # semantic goal encoder
        goal_vec =Variable(torch.LongTensor([i for i in range(1,22)]))
        goal_vec = goal_vec.to(semantic_map_sum.device)
        semantic_goal = self.obj_categories_embedding(goal_vec)
        # print("semantic_goal: ", semantic_goal.shape) # 1 21 256

        priors = torch.stack([torch.cat([semantic_map_encoder[i], semantic_goal],dim=1) for i in range(semantic_map_encoder.shape[0])])
        # print("priors: ", priors.shape) # torch.Size([2, 21, 512])

        # edge 
        # self.edge_index = torch.stack([self.edge for i in range(semantic_map_encoder.shape[0])])
        graph = torch.stack([self.graphcnn(
            Data(x = priors[i], 
                edge_index = self.edge, 
                edge_type = self.edge_type
                ).to(semantic_map_sum.device)
        ) for i in range(semantic_map_encoder.shape[0])])
        graph = graph.to(semantic_map_sum.device)
        # data = Data(x = priors, edge_index = self.edge_index)
        # print("gcn: ", graph.shape) # torch.Size([2, 21, 21])

        graph = graph.view(graph.size(0), -1)
        # print("graph: ", graph.shape) # 2 441

        graph = self.graph_fc(graph)
        # print("graph: ", graph.shape) # 2 256

        x.append(graph)

        curr_pose = observations["curr_pose"]
        curr_pose = curr_pose/map_sum.shape[2]
        # print("curr_pose: ", curr_pose)
        curr_pose_obs = self.curr_pose_embedding(curr_pose)

        x.append(curr_pose_obs)

        x = torch.cat(x, dim=1)

        x = nn.ReLU()(self.linear1(x))

        x = nn.ReLU()(self.linear2(x))

        return x


class SemanticMap_Encoder(nn.Module):
    def __init__(self,
        input_channels,
        output_channels,
    ):
        super(SemanticMap_Encoder, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(input_channels, 8, 7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)

        self.fc = nn.Linear(128, output_channels)

    def forward(self,x):
        # 21 1 240 240
        x = self.maxpool(x) # 21 1 120 120
        x = self.conv1(x) # 21 8 30 30
        x = nn.ReLU()(x)

        x = self.maxpool(x) # 21 8 15 15
        x = self.conv2(x) # 21 16 8 8
        x = nn.ReLU()(x)

        x = self.maxpool(x) # 21 16 4 4 
        x = self.conv3(x) # 21 32 2 2 
        x = nn.ReLU()(x)

        x = Flatten()(x.contiguous()) # 21 128

        x = self.fc(x) # 21 256

        return x
