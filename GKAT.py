import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.cuda.amp import autocast
from typing import Optional, Tuple, Union, List
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm

class GKATConfig:
    def __init__(self, in_channels=256, out_channels=512, heads=8, 
                 dropout=0.6, walk_length=4, use_cached=False, 
                 cache_path='./gkat_masks', use_cuda=True, use_amp=True,
                 kernel_type='random_walk', alpha=0.1, beta=0.1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.walk_length = walk_length
        self.use_cached = use_cached
        self.cache_path = cache_path
        self.use_cuda = use_cuda
        self.use_amp = use_amp
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.beta = beta

class GraphKernel:
    """图核计算类"""
    @staticmethod
    def random_walk_kernel(adj: torch.Tensor, length: int) -> torch.Tensor:
        """随机游走核"""
        kernel = torch.eye(adj.size(0), device=adj.device)
        result = kernel.clone()
        for _ in range(length):
            kernel = torch.mm(kernel, adj)
            result = result + kernel
        return result

    @staticmethod
    def diffusion_kernel(adj: torch.Tensor, alpha: float) -> torch.Tensor:
        """扩散核"""
        return torch.matrix_exp(alpha * adj)

    @staticmethod
    def p_step_kernel(adj: torch.Tensor, p: int) -> torch.Tensor:
        """p步核"""
        return torch.matrix_power(adj, p)

class GKATMaskGenerator(nn.Module):
    """GKAT掩码生成器"""
    def __init__(self, config: GKATConfig):
        super(GKATMaskGenerator, self).__init__()
        self.config = config
        self.cached_masks = {}
        
        if config.use_cached and config.cache_path:
            try:
                import os
                if os.path.exists(config.cache_path):
                    self.cached_masks = torch.load(config.cache_path)
            except Exception as e:
                print(f"Error loading masks from {config.cache_path}: {e}")
    
    def compute_kernel(self, adj: torch.Tensor) -> torch.Tensor:
        """计算图核"""
        if self.config.kernel_type == 'random_walk':
            return self.random_walk_kernel(adj, self.config.walk_length)
        elif self.config.kernel_type == 'diffusion':
            return self.diffusion_kernel(adj, self.config.alpha)
        elif self.config.kernel_type == 'p_step':
            return self.p_step_kernel(adj, self.config.walk_length)
        else:
            raise ValueError(f"Unknown kernel type: {self.config.kernel_type}")
    
    def random_walk_kernel(self, adj: torch.Tensor, length: int) -> torch.Tensor:
        """随机游走核"""
        kernel = torch.eye(adj.size(0), device=adj.device)
        result = kernel.clone()
        for _ in range(length):
            kernel = torch.mm(kernel, adj)
            result = result + kernel
        return result
    
    def diffusion_kernel(self, adj: torch.Tensor, alpha: float) -> torch.Tensor:
        """扩散核"""
        return torch.matrix_exp(alpha * adj)
    
    def p_step_kernel(self, adj: torch.Tensor, p: int) -> torch.Tensor:
        """p步核"""
        return torch.matrix_power(adj, p)
    
    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if self.config.use_cuda:
            edge_index = edge_index.cuda()
        
        edge_index = edge_index.to(torch.long)
        
        with torch.amp.autocast('cuda', enabled=self.config.use_cuda and self.config.use_amp):
            if self.config.use_cached:
                key = f"{edge_index.shape[1]}_{num_nodes}_{self.config.kernel_type}"
                if key in self.cached_masks:
                    return self.cached_masks[key]
            
            # 使用稀疏矩阵操作
            adj = torch.sparse_coo_tensor(
                edge_index,
                torch.ones(edge_index.size(1), device=edge_index.device),
                (num_nodes, num_nodes)
            )
            
            # 添加自环
            eye = torch.sparse_coo_tensor(
                torch.arange(num_nodes, device=edge_index.device).repeat(2, 1),
                torch.ones(num_nodes, device=edge_index.device),
                (num_nodes, num_nodes)
            )
            
            adj = adj + eye
            
            # 计算图核
            kernel = self.compute_kernel(adj.to_dense())
            
            # 归一化
            kernel = F.normalize(kernel, p=2, dim=1)
            
            if self.config.use_cached and self.config.cache_path:
                self.cached_masks[key] = kernel
                try:
                    torch.save(self.cached_masks, self.config.cache_path)
                except Exception as e:
                    print(f"Failed to save masks to {self.config.cache_path}: {e}")
            
            return kernel

class GKATLayer(MessagePassing):
    """GKAT层实现"""
    def __init__(self, config: GKATConfig):
        super(GKATLayer, self).__init__(aggr='add')
        self.config = config
        
        # 线性变换
        self.lin = nn.Linear(config.in_channels, config.heads * config.out_channels, bias=False)
        
        # 注意力参数
        self.att = nn.Parameter(torch.Tensor(1, config.heads, 2 * config.out_channels))
        
        # 掩码生成器
        self.mask_generator = GKATMaskGenerator(config)
        
        # 初始化参数
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        
        if config.use_cuda:
            self.cuda()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 确保输入在正确的设备上
        if self.config.use_cuda:
            x = x.cuda()
            edge_index = edge_index.cuda()
            if edge_attr is not None:
                edge_attr = edge_attr.cuda()
        
        # 线性变换
        x = self.lin(x).view(-1, self.config.heads, self.config.out_channels)
        
        # 生成掩码
        mask = self.mask_generator(edge_index, x.size(0))
        
        # 消息传递
        out = self.propagate(edge_index, x=x, mask=mask, edge_attr=edge_attr)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                index: torch.Tensor, mask: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 计算注意力系数
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # 应用掩码
        source, target = index[0], index[1]
        mask_values = mask[source, target].unsqueeze(1)
        alpha = alpha * mask_values
        
        # 注意力机制
        alpha = F.softmax(alpha, dim=0)
        alpha = F.dropout(alpha, p=self.config.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out.mean(dim=1)

class GKATNet(nn.Module):
    """完整的GKAT网络"""
    def __init__(self, config: GKATConfig, num_layers: int = 2):
        super(GKATNet, self).__init__()
        self.config = config
        self.num_layers = num_layers
        
        # 创建GKAT层
        self.layers = nn.ModuleList([
            GKATLayer(config) for _ in range(num_layers)
        ])
        
        # 跳跃连接
        self.skip = nn.Linear(config.in_channels, config.out_channels)
        
        if config.use_cuda:
            self.cuda()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 保存输入特征用于跳跃连接
        x_skip = self.skip(x)
        
        # 逐层处理
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # 添加跳跃连接
        x = x + x_skip
        
        return x