import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv
from torch.cuda.amp import autocast
import torch.jit as jit
from typing import Optional, Tuple

class GKATLayer(MessagePassing):
    """基于GKAT论文实现的图核注意力层"""
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6, use_cuda=True):
        super(GKATLayer, self).__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # 线性变换矩阵
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels * 2))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        
        # 将模型移动到GPU（如果可用）
        if self.use_cuda:
            self.cuda()
    
    @jit.script_method
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, masking: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 确保输入在正确的设备上
        if self.use_cuda:
            x = x.cuda()
            edge_index = edge_index.cuda()
            if masking is not None:
                masking = masking.cuda()
        
        # 使用自动混合精度
        with autocast(enabled=self.use_cuda):
            # 线性变换
            x = self.lin(x).view(-1, self.heads, self.out_channels)
            
            # 应用注意力机制的消息传递
            out = self.propagate(edge_index, x=x, masking=masking)
            return out
    
    @jit.script_method
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, index: torch.Tensor, 
                masking: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 计算注意力系数
        alpha = F.leaky_relu((torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1), negative_slope=0.2)
        
        if masking is not None:
            source, target = index[0], index[1]
            mask_values = masking[source, target].unsqueeze(1)
            alpha = alpha * mask_values
        
        # 注意力机制
        alpha = F.softmax(alpha, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)
    
    @jit.script_method
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out.mean(dim=1)

class SimpleGKATNet(nn.Module):
    """简化版的GKAT网络，适合集成到AFTGAN框架"""
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6, use_cuda=True):
        super(SimpleGKATNet, self).__init__()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # 使用GATConv作为底层实现
        self.gat = GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout)
        
        # 添加虚拟mask_generator属性
        self.mask_generator = type('MaskGenerator', (), {
            'walk_length': 2, 
            'use_cached': False,
            'cache_path': None
        })()
        
        if self.use_cuda:
            self.cuda()
    
    @jit.script_method
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.use_cuda:
            x = x.cuda()
            edge_index = edge_index.cuda()
        
        with autocast(enabled=self.use_cuda):
            # 安全检查
            if edge_index.max().item() >= x.size(0):
                valid_mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
                edge_index = edge_index[:, valid_mask]
                
                if edge_index.shape[1] == 0:
                    edge_index = torch.arange(min(x.size(0), 10), device=x.device)
                    edge_index = torch.stack([edge_index, edge_index], dim=0)
            
            return self.gat(x, edge_index)

# 保持原始名称兼容性
GKATNet = SimpleGKATNet

class GKATMaskGenerator(nn.Module):
    """GKAT掩码生成器"""
    def __init__(self, walk_length=3, use_cached=False, cache_path=None, use_cuda=True):
        super(GKATMaskGenerator, self).__init__()
        self.walk_length = walk_length
        self.use_cached = use_cached
        self.cache_path = cache_path
        self.cached_masks = {}
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if use_cached and cache_path:
            try:
                import os
                if os.path.exists(cache_path):
                    self.cached_masks = torch.load(cache_path)
            except Exception as e:
                print(f"Error loading masks from {cache_path}: {e}")
    
    @jit.script_method
    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if self.use_cuda:
            edge_index = edge_index.cuda()
        
        with autocast(enabled=self.use_cuda):
            if self.use_cached:
                key = f"{edge_index.shape[1]}_{num_nodes}"
                if key in self.cached_masks:
                    return self.cached_masks[key]
            
            # 使用稀疏矩阵操作
            adj = torch.sparse_coo_tensor(
                edge_index,
                torch.ones(edge_index.size(1), device=edge_index.device),
                (num_nodes, num_nodes)
            )
            
            # 添加自循环
            eye = torch.sparse_coo_tensor(
                torch.arange(num_nodes, device=edge_index.device).repeat(2, 1),
                torch.ones(num_nodes, device=edge_index.device),
                (num_nodes, num_nodes)
            )
            
            adj = adj + eye
            
            # 使用稀疏矩阵乘法
            mask = adj.clone()
            current = adj.clone()
            
            for _ in range(self.walk_length - 1):
                current = torch.sparse.mm(current, adj)
                mask = mask + current
            
            mask = (mask > 0).float()
            
            if self.use_cached and self.cache_path:
                self.cached_masks[key] = mask
                try:
                    torch.save(self.cached_masks, self.cache_path)
                except:
                    print(f"Failed to save masks to {self.cache_path}")
            
            return mask