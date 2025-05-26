import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import GKATConfig
from GKAT import SimpleGKATNet
from distributed_training import run_distributed_training
from example_config import create_distributed_config

def create_dummy_data(num_samples=1000, num_features=64, num_classes=10):
    """创建虚拟数据集"""
    x = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return x, y

def main():
    # 创建分布式配置
    config = create_distributed_config()
    
    # 创建虚拟数据集
    x_train, y_train = create_dummy_data()
    x_test, y_test = create_dummy_data(num_samples=200)
    
    # 创建数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # 定义模型创建函数
    def create_model(config):
        return SimpleGKATNet(config)
    
    # 定义优化器创建函数
    def create_optimizer(params):
        return optim.Adam(params, lr=config.learning_rate)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 运行分布式训练
    run_distributed_training(
        config=config,
        model_fn=create_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer_fn=create_optimizer
    )

if __name__ == "__main__":
    main() 