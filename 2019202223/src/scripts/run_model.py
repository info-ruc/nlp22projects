import numpy as np
import pandas as pd
from scipy import spatial
import stanza
import torch
from torch import nn, optim
import torch.nn.functional as F
import dgl 
from scipy.sparse import csr_matrix
from nltk.tokenize import RegexpTokenizer
import argparse

from utils import Get_train_dataset, Get_test_dataset, Get_wordembedding, Sentence_to_matrix, Matrix_to_graph, Sentence_to_matrix_lstm
from utils import Get_score_gcn
from model import BiLSTM_GCN_1, BiLSTM_GCN_2, BiLSTM

# Training settings
# hidden_lstm, hidden_gcn
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')

parser.add_argument('--hidden_lstm', type=int, default=100,
                    help='Number of hidden units for bilstm.')
parser.add_argument('--hidden_gcn', type=int, default=64,
                    help='Number of hidden units for gcn.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='each epoch.')
parser.add_argument('--gcn_layer', type=int, default=2,
                    help='Number of gcn layer.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load dataset

df_dataset_train = Get_train_dataset()
df_dataset_test = Get_test_dataset()
embedding_dict = Get_wordembedding()

senten_1,senten_2,syntax_1,syntax_2 = Sentence_to_matrix(df_dataset_train, embedding_dict)
senten_1_test,senten_2_test,syntax_1_test,syntax_2_test = Sentence_to_matrix(df_dataset_test, embedding_dict)
g1, g2 = Matrix_to_graph(senten_1,senten_2,syntax_1,syntax_2)
g1_test, g2_test = Matrix_to_graph(senten_1_test,senten_2_test,syntax_1_test,syntax_2_test)

# change label to one-hot encoding

labels = np.zeros((len(df_dataset_train), 2)).astype(int)
for i in range(len(df_dataset_train)):
    labels[i, int(df_dataset_train.iloc[i, 2])] = 1

labels_test = np.zeros((len(df_dataset_test), 2)).astype(int)
for i in range(len(df_dataset_test)):
    labels_test[i, int(df_dataset_test.iloc[i, 2])] = 1

# initialize model

print("\nnote: initialize model.\n")
if args.gcn_layer == 1:
    model = BiLSTM_GCN_1(300, args.hidden_lstm, args.hidden_gcn, 2)
elif args.gcn_layer == 2:
    model = BiLSTM_GCN_2(300, args.hidden_lstm, args.hidden_gcn, 2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

print("Finish initializing model.\n")

# begin training

print("\nBegin traing:\n")
model.train()
epoch_losses = []
nums_graph = args.batch_size
for epoch in range(args.epochs): 
    epoch_loss = 0
    for i in range(0, int(len(df_dataset_train)/nums_graph)):
        las = nums_graph * i
        
        g_1 = dgl.batch(g1[las:las+nums_graph])
        g_2 = dgl.batch(g2[las:las+nums_graph])
        label = torch.FloatTensor(labels[las:las+nums_graph])
            
            
        prediction = model(g_1, g_2)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss = epoch_loss/len(df_dataset_train)*nums_graph
    
    if (epoch+1) % 5 == 0:
        f1,acc = Get_score_gcn(model, df_dataset_test, nums_graph, g1_test, g2_test, labels_test)
        print('Epoch {}, loss {:.8f}'.format(epoch+1, epoch_loss))
    epoch_losses.append(epoch_loss)

# begin testing

model.eval()
f1,acc = Get_score_gcn(model, df_dataset_test, nums_graph, g1_test, g2_test, labels_test)
print("\n--------------------------------------------------------------")
print("Final result: f1 {:.8f} acc {:.8f}".format(f1, acc))
print("--------------------------------------------------------------")