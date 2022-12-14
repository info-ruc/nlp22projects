from scipy.stats import pearsonr
import gzip
import pandas as pd
from data import Data, DataLoader
from model import SCIModel
from constant import *
import os
import torch
from tqdm import tqdm
import argparse

FLAG_SIMILARITY = 0.5
STOCK_DATA_PATH = './data/stock_data.xlsx'

def read_gz(filepath):
    with gzip.open(filepath, 'r') as f:
        return [line.decode('utf-8').strip() for line in f]

def calCorr(data_dir, stock2id):
    corr = {}
    stocks = read_gz(data_dir + 'stock.txt.gz')
    stocks = sorted(stocks)

    s_data = pd.read_excel(STOCK_DATA_PATH)
    stock_datas = {}
    for stock_id in stocks:
        if stock_id in stock2id:
            t = s_data[s_data.stock == int(stock_id)].change_ratio.tolist()
            if len(t)>0:
                stock_datas[stock_id] = t
    
    for i in tqdm(range(len(stocks)-1), 'cal corr'):
        for j in range(i+1,len(stocks)):
            x, y = stocks[i], stocks[j]
            if x in stock_datas and y in stock_datas and len(stock_datas[y]) == len(stock_datas[x]) and len(stock_datas[y]) >= 2:
                corr[(x, y)] = (pearsonr(stock_datas[x], stock_datas[y])[0] >= FLAG_SIMILARITY)
    return corr

def fintest(args):
    dataset = Data(args.data_dir, set_name = 'test')
    model = SCIModel(args, dataset)
    now_epochs = 10 # args.epochs
    print("Now epoch: {}".format(now_epochs))
    pretrain_sd = torch.load('{}/transe_model_sd_epoch{}.ckpt'.format(args.output_dir, now_epochs))
    model.load_state_dict(pretrain_sd,False)

    all_stock_embed = model.stock(torch.tensor(range(0, dataset.stock.size)))
    model_corr = torch.cosine_similarity(all_stock_embed.unsqueeze(1),all_stock_embed.unsqueeze(0),dim=-1)
    if_rela = (model_corr >= FLAG_SIMILARITY).tolist()

    stock2id = {}
    i = 0
    for stock in dataset.stock.vocab:
        stock2id[stock] = i
        i += 1
    corr = calCorr(args.data_dir, stock2id)

    x, y = 0, 0
    for key in tqdm(corr,'comparing'):
        stock_i, stock_j = key
        i, j = stock2id[stock_i], stock2id[stock_j]
        if corr[key] == if_rela[i][j]:
            x += 1
        y += 1
    print('compare precision: ', x/y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default = './output/', help='One of {BEUTY, CD, CELL, CLOTHING}.')
    parser.add_argument('--output_dir', type=str, default='./tmp2/', help='model name.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size.')
    parser.add_argument('--embed_size', type=int, default=300, help='knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.')
    parser.add_argument('--device', type=int, default=400, help='batch size.')
    parser.add_argument('--l2_lambda', type=float, default=0.005, help='l2 lambda')
    parser.add_argument('--hidden_size', type=int, default=100, help='The hidden size of RNN')
    args = parser.parse_args()
    args.device = 'cpu'

    fintest(args)

    



    