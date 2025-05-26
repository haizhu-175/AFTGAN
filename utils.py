import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, InMemoryDataset
from sklearn.metrics import hamming_loss
from sklearn import metrics
from typing import Optional, Tuple, List, Dict, Any, Union
from GKAT import GKATConfig


def print_file(str_, save_file_path=None):
    print(str_)
    if save_file_path is not None:
        with open(save_file_path, 'a') as f:
            print(str_, file=f)
# def comuter_hammingloss(y_true,y_pred):
#     y_hot = np.array(y_pred>0.5,dtype=float)
#     HammingLoss =[]
#     for i in range()
class Metrictor_PPI:
    def __init__(self, pre_y, truth_y, is_binary=False):
        # 确保输入是numpy数组
        if isinstance(pre_y, torch.Tensor):
            pre_y = pre_y.cpu().numpy()
        if isinstance(truth_y, torch.Tensor):
            truth_y = truth_y.cpu().numpy()
            
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        
        # 添加异常处理
        try:
            self.auc = metrics.roc_auc_score(truth_y, pre_y)
        except ValueError:
            # 如果只有一个类别，将AUC设为0或其他默认值
            self.auc = 0.0
        
        self.hmloss = hamming_loss(truth_y, pre_y)

        if is_binary:
            length = pre_y.shape[0]
            for i in range(length):
                if pre_y[i] == truth_y[i]:
                    if truth_y[i] == 1:
                        self.TP += 1
                    else:
                        self.TN += 1
                elif truth_y[i] == 1:
                    self.FN += 1
                elif pre_y[i] == 1:
                    self.FP += 1
            self.num = length
        else:
            N, C = pre_y.shape
            for i in range(N):
                for j in range(C):
                    if pre_y[i][j] == truth_y[i][j]:
                        if truth_y[i][j] == 1:
                            self.TP += 1
                        else:
                            self.TN += 1
                    elif truth_y[i][j] == 1:
                        self.FN += 1
                    elif pre_y[i][j] == 0:
                        self.FP += 1
            self.num = N * C

    
    def show_result(self, is_print=False, file=None):
        self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
        self.Precision = self.TP / (self.TP + self.FP + 1e-10)
        self.Recall = self.TP / (self.TP + self.FN + 1e-10)
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)

        if is_print:
            print_file(f"Accuracy: {self.Accuracy:.4f}", file)
            print_file(f"Precision: {self.Precision:.4f}", file)
            print_file(f"Recall: {self.Recall:.4f}", file)
            print_file(f"F1-Score: {self.F1:.4f}", file)

class UnionFindSet(object):
    def __init__(self, m):
        # m, n = len(grid), len(grid[0])
        self.roots = [i for i in range(m)]
        self.rank = [0 for i in range(m)]
        self.count = m
        
        for i in range(m):
            self.roots[i] = i
 
    def find(self, member):
        tmp = []
        while member != self.roots[member]:
            tmp.append(member)
            member = self.roots[member]
        for root in tmp:
            self.roots[root] = member
        return member
        
    def union(self, p, q):
        parentP = self.find(p)
        parentQ = self.find(q)
        if parentP != parentQ:
            if self.rank[parentP] > self.rank[parentQ]:
                self.roots[parentQ] = parentP
            elif self.rank[parentP] < self.rank[parentQ]:
                self.roots[parentP] = parentQ
            else:
                self.roots[parentQ] = parentP
                self.rank[parentP] -= 1
            self.count -= 1


def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    """使用BFS获取子图"""
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    # 随机选择一个起始节点
    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)
        for edge_index in node_to_edge_index[cur_node]:
            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
    
    node_list = candiate_node + selected_node
    return selected_edge_index

def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    """使用DFS获取子图"""
    stack = []
    selected_edge_index = []
    selected_node = []

    # 随机选择一个起始节点
    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = stack[-1]
        if cur_node in selected_node:
            flag = True
            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1
                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]
                    
                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break
            if flag:
                stack.pop()
            continue
        else:
            selected_node.append(cur_node)
            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)
    
    return selected_edge_index

def get_device(config: GKATConfig) -> torch.device:
    """获取当前设备"""
    if config.use_cuda and torch.cuda.is_available():
        if config.distributed:
            return torch.device(f'cuda:{config.local_rank}')
        return torch.device('cuda')
    return torch.device('cpu')

def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   path: str,
                   config: GKATConfig):
    """保存模型检查点"""
    if config.distributed and config.local_rank != 0:
        return  # 只在主进程上保存
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'config': config.to_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: str) -> Tuple[nn.Module, torch.optim.Optimizer, int, float, GKATConfig]:
    """加载模型检查点"""
    checkpoint = torch.load(path)
    config = GKATConfig.from_dict(checkpoint['config'])
    
    # 创建模型和优化器
    model = SimpleGKATNet(config)
    optimizer = torch.optim.Adam(model.parameters())
    
    # 加载状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss'], config

def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
    """获取模型摘要"""
    from torchsummary import summary
    return str(summary(model, input_size))

def profile_model(model: nn.Module,
                 input_tensor: torch.Tensor,
                 num_runs: int = 100) -> Dict[str, Any]:
    """分析模型性能"""
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
    
    # 预热
    for _ in range(10):
        _ = model(input_tensor)
    
    # 测量时间
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        _ = model(input_tensor)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / num_runs
    
    # 测量内存
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        memory_cached = torch.cuda.max_memory_cached() / 1024**2  # MB
    else:
        memory_allocated = 0
        memory_cached = 0
    
    return {
        'average_time_ms': elapsed_time,
        'memory_allocated_mb': memory_allocated,
        'memory_cached_mb': memory_cached,
        'parameters': count_parameters(model)
    }

def create_optimizer(model: nn.Module, config: GKATConfig) -> torch.optim.Optimizer:
    """创建优化器"""
    optimizer_params = {
        'lr': config.learning_rate,
        'weight_decay': config.weight_decay
    }
    
    return torch.optim.Adam(model.parameters(), **optimizer_params)

def create_scheduler(optimizer: torch.optim.Optimizer,
                    config: GKATConfig) -> torch.optim.lr_scheduler._LRScheduler:
    """创建学习率调度器"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

def log_metrics(metrics: Dict[str, Any], step: int, writer=None):
    """记录指标"""
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
    
    # 打印指标
    print(f"Step {step}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

def create_data_loaders(train_dataset,
                       test_dataset,
                       config: GKATConfig) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    if config.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.local_rank,
            shuffle=True
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=config.world_size,
            rank=config.local_rank,
            shuffle=False
        )
    else:
        train_sampler = None
        test_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader