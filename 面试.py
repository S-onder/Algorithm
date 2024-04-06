# 手撕Attention
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
class SelfAttention(nn.Module):
    """自注意力机制"""
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.scale = 1 / math.sqrt(d_model)
        self.linear_q = nn.Linear(d_model, d_model, bias = True)
        self.linear_k = nn.Linear(d_model, d_model, bias = True)
        self.linear_v = nn.Linear(d_model, d_model, bias = True)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model #验证隐藏层维度
        q = self.linear_q(x) # (batch_size, seq_len, d_model)
        k = self.linear_k(x) # (batch_size, seq_len, d_model)
        v = self.linear_v(x) # (batch_size, seq_len, d_model)
        atten = torch.matmul(q, k.transpose(1,2)) * self.scale # (batch_size, seq_len, seq_len)
        atten = torch.softmax(atten, dim = -1) # (batch_size, seq_len, seq_len)
        score = torch.matmul(atten, v) # (batch_size, seq_len, d_model)
        return score, atten
    
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__() 
        self.d_model = d_model  
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.linear_q = nn.Linear(d_model, d_model, bias = True)
        self.linear_k = nn.Linear(d_model, d_model, bias = True)
        self.linear_v = nn.Linear(d_model, d_model, bias = True)
        self.scale = 1 / math.sqrt(d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model
        q = self.linear_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3) # (batch_size, n_head, seq_len, head_dim
        k = self.linear_k(x).reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3) # (batch_size, n_head, seq_len, head_dim
        v = self.linear_v(x).reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3) # (batch_size, n_head, seq_len, head_dim
        atten = torch.matmul(q, k.transpose(2, 3)) * self.scale # (batch_size, n_head, seq_len, seq_len)
        atten = torch.softmax(atten, dim = -1) # (batch_size, n_head, seq_len, seq_len)
        score = torch.matmul(atten, v) # (batch_size, n_head, seq_len, head_dim)
        score = score.permute(0, 2, 1, 3).reshape(batch_size, seq_len, d_model) # (batch_size, seq_len, d_model)
        return score, atten 

 # Normalization层   
class BatchNorm(nn.Module):
    """Batch Normalization"""
    def __init__(self, d_model, eps = 1e-6, training = True):
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.training = training
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        if self.is_training:
            mean = x.mean(dim = 0, keepdim = True)
            var = x.var(dim = 0, keepdim = True)
            x = (x - mean) / torch.sqrt(var + self.eps)
        y = self.gamma * x + self.beta
        return y
        
class LayerNorm(nn.Module):
    """Layer Normalization"""
    def __init__(self, d_model, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        y = self.gamma * x + self.beta
        return y
    
#AUC计算
def calculate_auc(y_true, y_scores):
    # 初始化变量
    pos = sum(y_true)
    neg = len(y_true) - pos
    auc = 0
    rank_sum = 0

    # 对预测分数和真实标签进行排序
    data = sorted(zip(y_scores, y_true), key=lambda x: x[0])

    # 计算AUC
    for i in range(len(data)):
        if data[i][1] == 1:
            rank_sum += i + 1
    auc = (rank_sum - (pos*(pos+1))/2) / (pos*neg)

    return auc

if __name__ == '__main__':
    x = torch.randn(2, 4, 512) #batch=2,seq=4,d_model=512
    # self_atten = SelfAttention(512)
    # score, atten = self_atten(x)
    # print(atten.sum(dim = -1))
    # linear = nn.Linear(512, 512, bias=True)
    # print(linear.weight.shape)
    # print(linear.bias.shape)
    # print(linear(x).shape)   
    import numpy as np
    import random
    import time
    from sklearn.metrics import roc_auc_score
    y_true = np.random.choice([0, 1], size=10000, p=[0.7, 0.3])
    y_pred = np.random.uniform(0, 1, 10000)
    # 计算并打印roc_auc_score的执行时间
    start = time.time()
    sklearn_auc = roc_auc_score(y_true, y_pred)
    end = time.time()
    print(f'roc_auc_score execution time: {end - start}, result : {sklearn_auc}')

    # 计算并打印calculate_auc的执行时间
    start = time.time()
    my_auc = calculate_auc(y_true, y_pred)
    end = time.time()
    print(f'calculate_auc execution time: {end - start}, result : {my_auc}')
