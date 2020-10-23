import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv
from habitat_baselines.common.utils import Flatten


class GraphCNN(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphCNN, self).__init__()
        self.conv1 = GCNConv(in_channels=in_c, out_channels=1024)
        self.conv2 = GCNConv(in_channels=1024, out_channels=512)
        self.conv3 = GCNConv(in_channels=512, out_channels=128)
        self.fc = nn.Linear(128, out_c)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index

        # print("x: ", x.shape) # torch.Size([2, 21, 512])
        # print("edge_index: ", edge_index.shape)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)


class GraphRCNN(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphRCNN, self).__init__()
        self.conv1 = RGCNConv(in_channels=in_c, out_channels=1024, num_relations=3)
        self.conv2 = RGCNConv(in_channels=1024, out_channels=512, num_relations=3)
        self.conv3 = RGCNConv(in_channels=512, out_channels=128, num_relations=3)
        self.fc = nn.Linear(128, out_c)

    def forward(self,data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        # print("x: ", x.shape) # torch.Size([2, 21, 512])
        # print("edge_index: ", edge_index.shape)
        # print("edge_type: ", edge_type.shape)

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)