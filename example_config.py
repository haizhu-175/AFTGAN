from config import GKATConfig
from GKAT import GKATLayer, SimpleGKATNet, GKATMaskGenerator
import torch

def create_distributed_config():
    """创建4卡分布式训练配置"""
    config = GKATConfig(
        in_channels=64,
        out_channels=32,
        heads=8,
        dropout=0.6,
        use_cuda=True,
        walk_length=3,
        use_cached=False,
        cache_path=None,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100,
        weight_decay=0.0001,
        distributed=True,  # 启用分布式训练
        world_size=4,     # 使用4个GPU
        local_rank=0,     # 本地rank，会在运行时自动设置
        backend='nccl',   # 使用NCCL后端
        use_amp=True     # 启用自动混合精度
    )
    
    # 验证配置
    config.validate()
    return config

def main():
    # 创建分布式配置
    config = create_distributed_config()
    
    # 创建模型实例
    gkat_layer = GKATLayer(config)
    gkat_net = SimpleGKATNet(config)
    mask_generator = GKATMaskGenerator(config)
    
    # 示例输入
    num_nodes = 10
    num_features = config.in_channels
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    ])
    
    # 使用模型
    if config.use_cuda:
        x = x.cuda()
        edge_index = edge_index.cuda()
    
    # 生成掩码
    mask = mask_generator(edge_index, num_nodes)
    
    # 前向传播
    with torch.no_grad():
        # 使用GKAT层
        out_layer = gkat_layer(x, edge_index, mask)
        print(f"GKATLayer output shape: {out_layer.shape}")
        
        # 使用GKAT网络
        out_net = gkat_net(x, edge_index)
        print(f"GKATNet output shape: {out_net.shape}")

if __name__ == "__main__":
    main() 