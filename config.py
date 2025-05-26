import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class GKATConfig:
    """GKAT模型配置类"""
    # 模型参数
    in_channels: int
    out_channels: int
    heads: int = 8
    dropout: float = 0.6
    use_cuda: bool = True
    
    # 掩码生成器参数
    walk_length: int = 3
    use_cached: bool = False
    cache_path: Optional[str] = None
    
    # 训练参数
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 0.0001
    
    # 分布式训练参数
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0
    backend: str = 'nccl'
    
    # 混合精度训练
    use_amp: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GKATConfig':
        """从字典创建配置实例"""
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")
    
    def validate(self):
        """验证配置参数的有效性"""
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.out_channels > 0, "out_channels must be positive"
        assert self.heads > 0, "heads must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.walk_length > 0, "walk_length must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.world_size > 0, "world_size must be positive"
        assert self.local_rank >= 0, "local_rank must be non-negative"
        
        if self.use_cuda and not torch.cuda.is_available():
            print("Warning: CUDA is not available, setting use_cuda to False")
            self.use_cuda = False
        
        if self.distributed and not torch.cuda.is_available():
            print("Warning: Distributed training requires CUDA, setting distributed to False")
            self.distributed = False 