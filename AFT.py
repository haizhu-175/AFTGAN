import numpy as np
import torch
from torch import nn
from torch.nn import init


class AFT_FULL(nn.Module):
    def __init__(self, d_model, n=36, simple=True):
        super(AFT_FULL, self).__init__()
        self.d_model = d_model
        self.n = n
        self.simple = simple
        self.fc_q = None
        self.fc_k = None
        self.fc_v = None
        self.position_biases = None
        self.initialized = False
        self.sigmoid = nn.Sigmoid()
        
    def _initialize_weights(self, actual_d_model, actual_n):
        # 初始化线性层
        self.fc_q = nn.Linear(actual_d_model, actual_d_model)
        self.fc_k = nn.Linear(actual_d_model, actual_d_model)
        self.fc_v = nn.Linear(actual_d_model, actual_d_model)
        
        # 初始化位置偏置
        if self.simple:
            self.position_biases = nn.Parameter(torch.zeros(actual_n, actual_n))
        else:
            self.position_biases = nn.Parameter(torch.ones(actual_n, actual_n))
        
        self.initialized = True
        print(f"AFT initialized with dimensions: d_model={actual_d_model}, n={actual_n}")
    
    def forward(self, input):
        bs, n, dim = input.shape
        
        # 如果尚未初始化或维度不匹配，则初始化权重
        if not self.initialized:
            self._initialize_weights(dim, n)
        
        # 确保输入和模型在同一设备上
        device = input.device
        self.to(device)
        
        # 分块处理长序列
        if n > self.n:
            return self._forward_chunked(input)
        return self._forward_original(input)

    def _forward_original(self, input):
        bs, n, dim = input.shape
        
        # 确保position_biases尺寸正确
        if n > self.position_biases.size(0):
            old_biases = self.position_biases
            new_biases = torch.zeros(n, n, device=input.device)
            new_biases[:old_biases.size(0), :old_biases.size(1)] = old_biases
            self.position_biases = nn.Parameter(new_biases)
            print(f"Position biases resized to {n}x{n}")
        
        q = self.fc_q(input)
        k = self.fc_k(input).view(1, bs, n, dim)
        v = self.fc_v(input).view(1, bs, n, dim)
        
        # 确保设备一致
        position_biases = self.position_biases.view(n, 1, -1, 1)
        
        # 数值稳定性优化
        max_val = torch.max(k + position_biases, dim=2, keepdim=True)[0]
        exp_val = torch.exp(k + position_biases - max_val)
        
        numerator = torch.sum(exp_val * v, dim=2)
        denominator = torch.sum(exp_val, dim=2)
        
        out = numerator / denominator
        return self.sigmoid(q) * out.permute(1, 0, 2)

    def _forward_chunked(self, input):
        bs, n, dim = input.shape
        q = self.fc_q(input)
        
        outputs = []
        for i in range(0, n, self.n):
            chunk = input[:, i:i+self.n, :]
            k = self.fc_k(chunk).view(1, bs, -1, dim)
            v = self.fc_v(chunk).view(1, bs, -1, dim)
            
            # 处理局部位置偏置
            chunk_size = chunk.size(1)
            bias = self.position_biases[i:i+chunk_size, i:i+chunk_size]
            bias = bias.view(chunk_size, 1, -1, 1)
            
            max_val = torch.max(k + bias, dim=2, keepdim=True)[0]
            exp_val = torch.exp(k + bias - max_val)
            
            numerator = torch.sum(exp_val * v, dim=2)
            denominator = torch.sum(exp_val, dim=2)
            outputs.append(numerator / denominator)
        
        out = torch.cat(outputs, dim=0).permute(1, 0, 2)
        return self.sigmoid(q) * out


