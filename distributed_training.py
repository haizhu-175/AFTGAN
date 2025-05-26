import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Callable, Optional
from config import GKATConfig

def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """设置分布式训练环境"""
    # 设置环境变量
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

class DistributedTrainer:
    """分布式训练器"""
    def __init__(self, config: GKATConfig):
        self.config = config
        self.rank = config.local_rank
        self.world_size = config.world_size
        
        if config.distributed:
            setup_distributed(self.rank, self.world_size, config.backend)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config.distributed:
            cleanup_distributed()
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """包装模型以支持分布式训练"""
        if self.config.distributed:
            model = model.to(self.rank)
            model = DDP(model, device_ids=[self.rank])
        return model
    
    def create_sampler(self, dataset) -> Optional[DistributedSampler]:
        """创建分布式采样器"""
        if self.config.distributed:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        return None
    
    def train(self, 
              model: torch.nn.Module,
              train_loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              criterion: Callable,
              num_epochs: int,
              device: torch.device):
        """执行分布式训练"""
        model = self.wrap_model(model)
        model.train()
        
        for epoch in range(num_epochs):
            if self.config.distributed:
                train_loader.sampler.set_epoch(epoch)
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0 and self.rank == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}')
    
    def evaluate(self,
                model: torch.nn.Module,
                test_loader: torch.utils.data.DataLoader,
                criterion: Callable,
                device: torch.device) -> float:
        """执行分布式评估"""
        model = self.wrap_model(model)
        model.eval()
        
        total_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_loss += criterion(output, target).item()
        
        if self.config.distributed:
            # 在所有进程间同步损失
            dist.all_reduce(torch.tensor(total_loss).to(device))
            total_loss /= self.world_size
        
        return total_loss / len(test_loader)

def run_distributed_training(config: GKATConfig,
                           model_fn: Callable,
                           train_loader: torch.utils.data.DataLoader,
                           test_loader: torch.utils.data.DataLoader,
                           criterion: Callable,
                           optimizer_fn: Callable):
    """运行分布式训练的主函数"""
    if config.distributed:
        mp.spawn(
            _distributed_worker,
            args=(config, model_fn, train_loader, test_loader, criterion, optimizer_fn),
            nprocs=config.world_size,
            join=True
        )
    else:
        _distributed_worker(0, config, model_fn, train_loader, test_loader, criterion, optimizer_fn)

def _distributed_worker(rank: int,
                       config: GKATConfig,
                       model_fn: Callable,
                       train_loader: torch.utils.data.DataLoader,
                       test_loader: torch.utils.data.DataLoader,
                       criterion: Callable,
                       optimizer_fn: Callable):
    """分布式训练的工作进程"""
    config.local_rank = rank
    device = torch.device(f'cuda:{rank}' if config.use_cuda else 'cpu')
    
    with DistributedTrainer(config) as trainer:
        # 创建模型
        model = model_fn(config).to(device)
        
        # 创建优化器
        optimizer = optimizer_fn(model.parameters())
        
        # 训练模型
        trainer.train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.num_epochs,
            device=device
        )
        
        # 评估模型
        if rank == 0:  # 只在主进程上评估
            test_loss = trainer.evaluate(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device
            )
            print(f'Test Loss: {test_loss:.6f}') 