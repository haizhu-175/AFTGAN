import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from torch.amp import autocast, GradScaler

from gnn_data import GNN_DATA
from gnn_model import GIN_Net2
from GKAT import GKATNet
# from model import GIN_Net2
from utils import Metrictor_PPI, print_file

from tensorboardX import SummaryWriter

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def boolean_string(s):
    # if s not in {'False', 'True'}:
    #     raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--description', default=None, type=str,
                    help='train description')
parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default=None, type=str,
                    help='protein sequence vector path')
parser.add_argument('--split_new', default=None, type=boolean_string,
                    help='split new index file or not')
parser.add_argument('--split_mode', default=None, type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--train_valid_index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--use_lr_scheduler', default=None, type=boolean_string,
                    help="train use learning rate scheduler or not")
parser.add_argument('--save_path', default=None, type=str,
                    help='model save path')
parser.add_argument('--graph_only_train', default=None, type=boolean_string,
                    help='train ppi graph conctruct by train or all(train with test)')
parser.add_argument('--batch_size', default=None, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--epochs', default=None, type=int,
                    help='train epoch number')
parser.add_argument('--use_gkat', action='store_true',
                   help='使用GKAT替代GAT')
parser.add_argument('--walk_length', type=int, default=4,
                   help='GKAT的随机游走长度')
parser.add_argument('--use_cached_masks', action='store_true',
                   help='使用缓存的GKAT掩码')
parser.add_argument('--mask_cache_path', type=str, default='./gkat_masks',
                   help='GKAT掩码缓存路径')
parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练')
parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数')

def train(model, graph, ppi_list, loss_fn, optimizer, device,
        result_file_path, summary_writer, save_path,
        batch_size=512, epochs=1000, scheduler=None, 
        got=False, use_amp=False, grad_clip=1.0, accumulation_steps=1):
    
    # 初始化梯度缩放器
    scaler = GradScaler('cuda', enabled=use_amp)
    
    # 减小批次大小以适应GPU内存
    batch_size = min(batch_size, 32)
    
    # 确保所有数据在正确的设备上
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    if hasattr(graph, 'edge_attr_1'):
        graph.edge_attr_1 = graph.edge_attr_1.to(device)
    if hasattr(graph, 'edge_attr_got'):
        graph.edge_attr_got = graph.edge_attr_got.to(device)
    
    # 确保数据类型一致
    graph.x = graph.x.to(torch.float32)
    graph.edge_index = graph.edge_index.to(torch.long)
    if hasattr(graph, 'edge_attr_1'):
        graph.edge_attr_1 = graph.edge_attr_1.to(torch.float32)
    if hasattr(graph, 'edge_attr_got'):
        graph.edge_attr_got = graph.edge_attr_got.to(torch.float32)
    
    # 检查edge_index的有效性
    max_node_idx = graph.x.size(0) - 1
    edge_max = graph.edge_index.max().item()
    if edge_max > max_node_idx:
        print(f"Warning: edge_index contains indices ({edge_max}) larger than the number of nodes ({max_node_idx})")
        print("Attempting to fix edge_index...")
        valid_edges = (graph.edge_index[0] <= max_node_idx) & (graph.edge_index[1] <= max_node_idx)
        graph.edge_index = graph.edge_index[:, valid_edges]
        if hasattr(graph, 'edge_attr_1'):
            graph.edge_attr_1 = graph.edge_attr_1[valid_edges]
        valid_train_mask = []
        for i, idx in enumerate(graph.train_mask):
            if idx < graph.edge_index.size(1):
                valid_train_mask.append(idx)
        graph.train_mask = valid_train_mask
        print(f"Updated edge_index shape: {graph.edge_index.shape}")
        print(f"Updated train_mask length: {len(graph.train_mask)}")
    
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0
    truth_edge_num = graph.edge_index.shape[1] // 2

    for epoch in range(epochs):
        # 内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)
        print(f"Steps per epoch: {steps}")

        model.train()
        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)

        optimizer.zero_grad()  # 在epoch开始时清零梯度

        for step in range(steps):
            if step == steps-1:
                if got:
                    train_edge_id = graph.train_mask_got[step*batch_size:]
                else:
                    train_edge_id = graph.train_mask[step*batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step*batch_size : step*batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step*batch_size : step*batch_size + batch_size]
            
            # 确保train_edge_id是列表或张量
            if isinstance(train_edge_id, list):
                train_edge_id = torch.tensor(train_edge_id, device=device, dtype=torch.long)
            else:
                train_edge_id = train_edge_id.to(device)
            
            # 使用自动混合精度
            with autocast('cuda', enabled=use_amp):
                if got:
                    output = model(graph.x, graph.edge_index_got, train_edge_id)
                    label = graph.edge_attr_got[train_edge_id]
                else:
                    output = model(graph.x, graph.edge_index, train_edge_id)
                    label = graph.edge_attr_1[train_edge_id]
                
                label = label.type(torch.float32).to(device)
                loss = loss_fn(output, label) / accumulation_steps  # 缩放损失

            # 使用梯度缩放器进行反向传播
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.float32).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data)
            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item() * accumulation_steps  # 恢复原始损失值

            summary_writer.add_scalar('train/loss', loss.item() * accumulation_steps, global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)
            summary_writer.add_scalar('train/auc', metrics.auc, global_step)
            summary_writer.add_scalar('train/hmloss', metrics.hmloss, global_step)

            global_step += 1
            print_file(f"epoch: {epoch}, step: {step}, Train: label_loss: {loss.item() * accumulation_steps}, "
                      f"precision: {metrics.Precision}, recall: {metrics.Recall}, f1: {metrics.F1}, "
                      f"auc: {metrics.auc}, hmloss: {metrics.hmloss}")
        
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }, os.path.join(save_path, 'gnn_model_train.ckpt'))
        
        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0

        model.eval()
        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps-1:
                    valid_edge_id = graph.val_mask[step*batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step*batch_size : step*batch_size + batch_size]
                
                # 确保valid_edge_id是列表或张量
                if isinstance(valid_edge_id, list):
                    valid_edge_id = torch.tensor(valid_edge_id, device=device, dtype=torch.long)
                else:
                    valid_edge_id = valid_edge_id.to(device)
                
                with autocast('cuda', enabled=use_amp):
                    output = model(graph.x, graph.edge_index, valid_edge_id)
                    label = graph.edge_attr_1[valid_edge_id]
                    label = label.type(torch.float32).to(device)
                    loss = loss_fn(output, label)
                
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.float32).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
        
        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)
        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps
        valid_loss = valid_loss_sum / valid_steps

        if scheduler is not None:
            scheduler.step(loss)
            print_file(f"epoch: {epoch}, now learning rate: {scheduler.optimizer.param_groups[0]['lr']}", 
                      save_file_path=result_file_path)
        
        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict()
            }, os.path.join(save_path, 'gnn_model_valid_best.ckpt'))
        
        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)
        summary_writer.add_scalar('valid/auc', metrics.auc, global_step)
        summary_writer.add_scalar('valid/hmloss', metrics.hmloss, global_step)

        print_file(f"epoch: {epoch}, Training_avg: label_loss: {loss}, recall: {recall}, precision: {precision}, "
                  f"F1: {f1}, Validation_avg: loss: {valid_loss}, recall: {metrics.Recall}, "
                  f"precision: {metrics.Precision}, F1: {metrics.F1}, auc: {metrics.auc}, "
                  f"hmloss: {metrics.hmloss}, Best valid_f1: {global_best_valid_f1}, in {global_best_valid_f1_epoch} epoch",
                  save_file_path=result_file_path)

def main():
    import os  # 在函数内部再次导入os模块，确保可用
    args = parser.parse_args()

    # 创建保存路径
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 设置GKAT掩码缓存路径
    if args.use_gkat and args.use_cached_masks:
        if not os.path.exists(args.mask_cache_path):
            os.makedirs(args.mask_cache_path, exist_ok=True)

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    print("use_get_feature_origin")
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)

    ppi_data.generate_data()

    print("----------------------- start split train and valid index -------------------")
    print("whether to split new train and valid index file, {}".format(args.split_new))
    if args.split_new:
        print("use {} method to split".format(args.split_mode))
    ppi_data.split_dataset(args.train_valid_index_path, random_new=args.split_new, mode=args.split_mode)
    print("----------------------- Done split train and valid index -------------------")

    graph = ppi_data.data
    print(f"Graph x shape: {graph.x.shape}")
    
    # 在这里定义 device 变量
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # 获取实际维度
    in_feature = 13  # 氨基酸向量维度固定为13
    in_len = graph.x.shape[1]  # 序列长度
    
    # 修改模型初始化
    model = GIN_Net2(
        in_len=in_len,
        in_feature=in_feature,  
        gin_in_feature=256, 
        num_layers=1, 
        pool_size=3, 
        cnn_hidden=1,
        feature_fusion='mul',  # 显式设置feature_fusion
        use_gkat=args.use_gkat,  # 传递使用GKAT的标志
        walk_length=args.walk_length  # 传递walk_length
    ).to(device)
    
    # 如果使用GKAT并需要缓存掩码
    if args.use_gkat and hasattr(model, 'graph_layer') and hasattr(model.graph_layer, 'mask_generator'):
        # 设置掩码生成器参数
        model.graph_layer.mask_generator.walk_length = args.walk_length
        model.graph_layer.mask_generator.use_cached = args.use_cached_masks
        
        if args.use_cached_masks:
            import os
            if not os.path.exists(args.mask_cache_path):
                os.makedirs(args.mask_cache_path, exist_ok=True)
            
            # 设置掩码缓存路径
            cache_file = os.path.join(args.mask_cache_path, f"gkat_masks_l{args.walk_length}.pt")
            model.graph_layer.mask_generator.cache_path = cache_file
            print(f"GKAT掩码将缓存于: {cache_file}")
    
    ppi_list = ppi_data.ppi_list
    # print(ppi_list.shape)
    # print(ppi_list)

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    graph.edge_index_got = torch.cat((graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    graph.edge_attr_got = torch.cat((graph.edge_attr_1[graph.train_mask], graph.edge_attr_1[graph.train_mask]), dim=0)
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(args.save_path, "gnn_{}_{}".format(args.description, time_stamp))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    os.mkdir(save_path)

    with open(config_path, 'w') as f:
        args_dict = args.__dict__
        for key in args_dict:
            f.write("{} = {}".format(key, args_dict[key]))
            f.write('\n')
        f.write('\n')
        f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    summary_writer = SummaryWriter(save_path)

    train(model, graph, ppi_list, loss_fn, optimizer, device,
        result_file_path, summary_writer, save_path,
        batch_size=args.batch_size, epochs=args.epochs, scheduler=scheduler, 
        got=args.graph_only_train,
        use_amp=args.use_amp, grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps)
    
    summary_writer.close()


if __name__ == "__main__":
    main()
