import os
import numpy as np
import random
import torch
from sklearn.metrics import hamming_loss
from sklearn import metrics


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
                    elif truth_y[i][j] == 0:
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