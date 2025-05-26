import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


from EMSA import EMSA
from ExternalAttention import  ExternalAttention
from GKAT import GKATNet, GKATConfig  # 添加GKATConfig导入


from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv,GATConv


class GATNet(torch.nn.Module):
    def __init__(self, num_feature, out_feature,him):
        super(GATNet, self).__init__()
        self.GAT1 = GATConv(num_feature, him, heads=8, concat=True, dropout=0.6)
        self.GAT2 = GATConv(8*him, out_feature, dropout=0.6)

    def forward(self, x,edge_index):
        # x, edge_index = data.x, data.edge_index

        x = self.GAT1(x, edge_index)
        x = F.relu(x)
        x = self.GAT2(x, edge_index)
        return x

class GIN_Net2(torch.nn.Module):
    def __init__(self, in_len=500, in_feature=13, gin_in_feature=256, num_layers=1,
                 hidden=512, use_jk=False, pool_size=3, cnn_hidden=1, train_eps=True,
                 feature_fusion='mul', class_num=7, use_gkat=False, walk_length=4,
                 gkat_heads=8, gkat_dropout=0.6, gkat_kernel_type='random_walk',
                 gkat_alpha=0.1, gkat_beta=0.1, gkat_num_layers=2):
        super(GIN_Net2, self).__init__()
        
        # 序列特征处理
        self.conv1d = nn.Conv1d(in_channels=in_feature, out_channels=cnn_hidden, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.fc1 = nn.Linear((in_len - 2) // pool_size, gin_in_feature)
        
        # 选择使用GKAT还是GAT
        if use_gkat:
            print(f"使用GKAT模块 (walk_length={walk_length}, kernel_type={gkat_kernel_type})")
            gkat_config = GKATConfig(
                in_channels=gin_in_feature,
                out_channels=hidden,
                heads=gkat_heads,
                dropout=gkat_dropout,
                walk_length=walk_length,
                kernel_type=gkat_kernel_type,
                alpha=gkat_alpha,
                beta=gkat_beta,
                use_cuda=torch.cuda.is_available(),
                use_amp=True
            )
            self.graph_layer = GKATNet(gkat_config, num_layers=gkat_num_layers)
        else:
            print("使用标准GAT模块")
            self.graph_layer = GATNet(gin_in_feature, hidden, 10)
            
        # 后续处理层
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.feature_fusion = 'mul' if feature_fusion is None else feature_fusion
        self.fc2 = nn.Linear(hidden if self.feature_fusion == 'mul' else hidden * 2, class_num)
        
    def forward(self, x, edge_index, train_edge_id, p=0.5):
        # 确保输入数据类型一致
        x = x.to(torch.float32)
        
        # 序列特征处理
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.maxpool1d(x)
        
        x = x.squeeze()
        x = self.fc1(x)
        
        # 安全检查：确保edge_index中的索引在有效范围内
        max_index = edge_index.max().item()
        if max_index >= x.size(0):
            print(f"警告: edge_index中存在无效索引 ({max_index} >= {x.size(0)})")
            # 过滤无效边
            valid_mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
            edge_index = edge_index[:, valid_mask]
            print(f"过滤后: edge_index.shape = {edge_index.shape}")
            
            # 检查并更新train_edge_id
            if isinstance(train_edge_id, list) or isinstance(train_edge_id, (torch.Tensor, np.ndarray)):
                valid_ids = []
                for idx in train_edge_id if isinstance(train_edge_id, list) else train_edge_id.tolist():
                    if idx < edge_index.shape[1]:
                        valid_ids.append(idx)
                if len(valid_ids) > 0:
                    train_edge_id = valid_ids
                else:
                    train_edge_id = list(range(min(10, edge_index.shape[1])))
        
        # 图特征提取
        x = self.graph_layer(x, edge_index)
        
        # 全连接层处理
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)
        x = self.lin2(x)
        
        # 提取边特征
        if edge_index.shape[1] > 0 and (isinstance(train_edge_id, list) and len(train_edge_id) > 0) or \
           (isinstance(train_edge_id, torch.Tensor) and train_edge_id.numel() > 0):
            node_id = edge_index[:, train_edge_id]
            x1 = x[node_id[0]]
            x2 = x[node_id[1]]
            
            # 特征融合
            if self.feature_fusion == 'concat':
                x = torch.cat([x1, x2], dim=1)
            else:  # mul
                x = torch.mul(x1, x2)
                
            x = self.fc2(x)
        else:
            # 如果没有有效边，返回零张量
            shape = [len(train_edge_id) if isinstance(train_edge_id, list) else train_edge_id.shape[0]]
            x = torch.zeros(shape + [self.fc2.weight.shape[1]], device=x.device, dtype=torch.float32)
        
        return x