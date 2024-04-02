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
    

if __name__ == '__main__':
    x = torch.randn(2, 4, 512) #batch=2,seq=4,d_model=512
    self_atten = SelfAttention(512)
    score, atten = self_atten(x)
    print(atten.sum(dim = -1))
