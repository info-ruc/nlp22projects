import numpy as np
import pandas as pd
from scipy import spatial
import stanza
from nltk.tokenize import RegexpTokenizer
import torch
from torch import nn, optim
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset
from dgl.nn.pytorch import GraphConv
from scipy.sparse import csr_matrix

class BiLSTM_GCN_2(nn.Module):
    def __init__(self, input_size, hidden_lstm, hidden_gcn, output_size):
        super(BiLSTM_GCN_2, self).__init__()
            
        self.input_size = input_size
        self.hidden_lstm = hidden_lstm
        self.hidden_gcn = hidden_gcn
        self.output_size = output_size
            
        # BiLSTM Layer
        self.rnn_1 = nn.LSTM(input_size, hidden_lstm, bidirectional=True, batch_first=True, dropout=0.5, num_layers=2)
        self.rnn_2 = nn.LSTM(input_size, hidden_lstm, bidirectional=True, batch_first=True, dropout=0.5, num_layers=2)
        self.linear = nn.Linear(hidden_lstm * 2, hidden_gcn)
        self.dropout = nn.Dropout(0.5)
            
        # GCN Layer - 1
        self.conv1_1 = GraphConv(2*hidden_lstm, hidden_gcn, allow_zero_in_degree = True)
        self.conv1_2 = GraphConv(2*hidden_lstm, hidden_gcn, allow_zero_in_degree = True)
            
        # GCN Layer - 2
        self.conv2_1 = GraphConv(hidden_gcn, hidden_gcn, allow_zero_in_degree = True)
        self.conv2_2 = GraphConv(hidden_gcn, hidden_gcn, allow_zero_in_degree = True)
            
        # Predict Layer
        self.predict = nn.Linear(hidden_gcn, output_size)   # 定义分类器

    def forward(self, g_1, g_2):
        self.rnn_1.flatten_parameters()
        self.rnn_2.flatten_parameters()
            
        in_1 = g_1.ndata['x'].reshape(-1, 35, self.input_size)
        in_2 = g_2.ndata['x'].reshape(-1, 35, self.input_size)
            
        o_1, (h_1, c_1) = self.rnn_1(in_1)  
        o_2, (h_2, c_2) = self.rnn_2(in_2)
            
        # ------ size ------
        # in: batch_size * T * input_size 
        # o: batch_size * T * (2*hidden_lstm)
        # h: (2*self.num_layers(lstm的层数)) * batch_size * hidden_lstm

        # hls_1 = torch.cat([h_1[-1, :, :], h_1[-2, :, :]], dim=-1)
        # hls_2 = torch.cat([h_2[-1, :, :], h_2[-2, :, :]], dim=-1)
            
            
        # get context representation as word's feature
        wf_1 = o_1.reshape(-1, 2*self.hidden_lstm)
        wf_2 = o_2.reshape(-1, 2*self.hidden_lstm)
        # ------ size ------
        # wf: (batch_size * T) * (2*hidden_lstm) [2 dim]

        l2g_1 = F.relu(self.linear(wf_1))
        l2g_2 = F.relu(self.linear(wf_2))
            
        # get final representation by mean pool
        layer_1_h_1 = F.relu(self.conv1_1(g_1, wf_1))  # [(batch_size * T), hidden_gcn]
        layer_1_h_1 = 0.6*layer_1_h_1 + 0.4*l2g_1
        g_1.ndata['h1'] = layer_1_h_1
        layer_2_h_1 = F.relu(self.conv2_1(g_1, layer_1_h_1))  # [(batch_size * T), hidden_gcn]
        g_1.ndata['h2'] = 0.6*layer_2_h_1 + 0.4*l2g_1
        # g_1.ndata['h2'] = layer_2_h_1
        final_rep_1 = dgl.mean_nodes(g_1, 'h2')  # [batch_size, hidden_gcn]
            
        layer_1_h_2 = F.relu(self.conv1_2(g_2, wf_2))  # [(batch_size * T), hidden_gcn]
        layer_1_h_2 = 0.6*layer_1_h_2 + 0.4*l2g_2
        g_2.ndata['h1'] = layer_1_h_2
        layer_2_h_2 = F.relu(self.conv2_2(g_2, layer_1_h_2))  # [(batch_size * T), hidden_gcn]
        g_2.ndata['h2'] = 0.6*layer_2_h_2 + 0.4*l2g_2
        # g_2.ndata['h2'] = layer_2_h_2
        final_rep_2 = dgl.mean_nodes(g_2, 'h2')   # [batch_size, hidden_gcn]
            
        # get distance between 2 sentence 
        dist = torch.tanh(final_rep_1 - final_rep_2)# [batch_size, hidden_gcn]
        
        return self.predict(dist)


class BiLSTM_GCN_1(nn.Module):
    def __init__(self, input_size, hidden_lstm, hidden_gcn, output_size):
        super(BiLSTM_GCN_1, self).__init__()
            
        self.input_size = input_size
        self.hidden_lstm = hidden_lstm
        self.hidden_gcn = hidden_gcn
        self.output_size = output_size
            
        # BiLSTM Layer
        self.rnn_1 = nn.LSTM(input_size, hidden_lstm, bidirectional=True, batch_first=True, dropout=0.5, num_layers=2)
        self.rnn_2 = nn.LSTM(input_size, hidden_lstm, bidirectional=True, batch_first=True, dropout=0.5, num_layers=2)
        self.dropout = nn.Dropout(0.5)
            
        # GCN Layer - 1
        self.conv1_1 = GraphConv(2*hidden_lstm, hidden_gcn, allow_zero_in_degree = True)
        self.conv1_2 = GraphConv(2*hidden_lstm, hidden_gcn, allow_zero_in_degree = True)
            
        # Predict Layer
        self.predict = nn.Linear(hidden_gcn, output_size)   # 定义分类器

    def forward(self, g_1, g_2):
        self.rnn_1.flatten_parameters()
        self.rnn_2.flatten_parameters()
            
        in_1 = g_1.ndata['x'].reshape(-1, 35, self.input_size)
        in_2 = g_2.ndata['x'].reshape(-1, 35, self.input_size)
            
        o_1, (h_1, c_1) = self.rnn_1(in_1)  
        o_2, (h_2, c_2) = self.rnn_2(in_2)
            
        # ------ size ------
        # in: batch_size * T * input_size 
        # o: batch_size * T * (2*hidden_lstm)
        # h: (2*self.num_layers(lstm的层数)) * batch_size * hidden_lstm

        # hls_1 = torch.cat([h_1[-1, :, :], h_1[-2, :, :]], dim=-1)
        # hls_2 = torch.cat([h_2[-1, :, :], h_2[-2, :, :]], dim=-1)
            
            
        # get context representation as word's feature
        wf_1 = o_1.reshape(-1, 2*self.hidden_lstm)
        wf_2 = o_2.reshape(-1, 2*self.hidden_lstm)
        # ------ size ------
        # wf: (batch_size * T) * (2*hidden_lstm) [2 dim]
            
        # get final representation by mean pool
        layer_1_h_1 = F.relu(self.conv1_1(g_1, wf_1))  # [(batch_size * T), hidden_gcn]
        g_1.ndata['h1'] = layer_1_h_1
        final_rep_1 = dgl.mean_nodes(g_1, 'h1')  # [batch_size, hidden_gcn]
            
        layer_1_h_2 = F.relu(self.conv1_2(g_2, wf_2))  # [(batch_size * T), hidden_gcn]
        g_2.ndata['h1'] = layer_1_h_2
        final_rep_2 = dgl.mean_nodes(g_2, 'h1')   # [batch_size, hidden_gcn]
            
        # get distance between 2 sentence 
        dist = torch.tanh(final_rep_1 - final_rep_2)# [batch_size, hidden_gcn]
        
        return self.predict(dist)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.rnn_1 = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True, num_layers=2, dropout=0.5)
        self.rnn_2 = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True, num_layers=2, dropout=0.5)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.5)
            

    def forward(self, in_1, in_2):

        self.rnn_1.flatten_parameters()
        self.rnn_2.flatten_parameters()
        o_1, (h_1, c_1) = self.rnn_1(in_1)  
        o_2, (h_2, c_2) = self.rnn_2(in_2)    
        # in: batch_size * T * input_size 
        # o: batch_size * T * (2*hidden_size)
        # h: (2*self.num_layers(lstm的层数)) * batch_size * hidden_size
            
        out1 = torch.cat([h_1[-1, :, :], h_1[-2, :, :]], dim=-1)
        out2 = torch.cat([h_2[-1, :, :], h_2[-2, :, :]], dim=-1)
        # out: batch_size * （2*hidden_size）
        #senten_out = torch.cat([out1, out2], dim=1)
            
        senten_out = out1 - out2
        senten_out = torch.tanh(senten_out)
        output = self.linear(senten_out)
        # output: batch_size * 2
        return output