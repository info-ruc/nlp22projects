from data import Data, DataLoader
from model import SCIModel
from constant import *
import os
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import gzip
import argparse
import statsmodels.api as sm

FLAG_SIMILARITY = 0.5
STOCK_DATA_PATH = './data/stock_data.xlsx'

def prepare_dataset(args):
    dataset = Data(args.data_dir, set_name = 'train')
    save_dataset(args.output_dir, dataset)
    print('save dataset done!!!')

def train(args):
    dataset = load_dataset(args.output_dir)
    model = SCIModel(args, dataset)

    dataloader = DataLoader(dataset, args.batch_size)
    posts_to_train = args.epochs * dataloader.post_size
    words_to_train = float(args.epochs * dataset.word_count) + 1
    print('posts to train size :', posts_to_train)
    
    now_epochs = 0#args.start_epoch
    ck_path = '{}/transe_model_sd_epoch{}.ckpt'.format(args.output_dir, now_epochs)
    if os.path.exists(ck_path):
        pretrain_sd = torch.load(ck_path)
        model.load_state_dict(pretrain_sd,False)
    else:
        now_epochs = 0
    model.to(args.device)

    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    steps = 0
    smooth_loss = 0.0

    for epoch in tqdm(range(now_epochs + 1, args.epochs + now_epochs + 1)):
        dataloader.reset()
        while dataloader.has_next():
            learning_rate = args.lr * max(0.0001, 
									1.0 - dataloader.finished_word_num / words_to_train)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # Get training batch.
            batch_idxs = dataloader.get_batch()

            # Train model.
            optimizer.zero_grad()
            train_loss = model(batch_idxs) # + reasoner((batch_idxs,query_idxs))
            train_loss.backward() # test_loss 测试
            optimizer.step()

            smooth_loss += train_loss.item() / args.steps_per_checkpoint
            
            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                logger.info('Epoch: {:02d} | '.format(epoch) +
                            'Reviews: {:d}/{:d} | '.format(dataloader.finished_post_num, posts_to_train) +
                            'Lr: {:.5f} | '.format(learning_rate) +
                            'Smooth loss: {:.5f}'.format(smooth_loss))
                smooth_loss = 0.0
        torch.save(model.state_dict(), '{}/transe_model_sd_epoch{}.ckpt'.format(args.output_dir, epoch))
    print("finish train")

def read_gz(filepath):
    with gzip.open(filepath, 'r') as f:
        return [line.decode('utf-8').strip() for line in f]

def calCorr(data_dir, stock2id):
    return_corr = {}
    corr = {}
    price_corr = {}
    stocks = read_gz(data_dir + 'stock.txt.gz')
    stocks = sorted(stocks)

    s_data = pd.read_excel(STOCK_DATA_PATH)
    stock_datas = {}
    price_datas = {}
    for stock_id in stocks:
        if stock_id in stock2id:
            t = s_data[s_data.stock == int(stock_id)].change_ratio.tolist()
            if len(t)>0:
                stock_datas[stock_id] = t
                price_datas[stock_id] = s_data[s_data.stock == int(stock_id)].clsprc.tolist()
    
    for i in tqdm(range(len(stocks)-1), 'cal corr'):
        for j in range(i+1,len(stocks)):
            x, y = stocks[i], stocks[j]
            if x in stock_datas and y in stock_datas and len(stock_datas[y]) == len(stock_datas[x]) and len(stock_datas[y]) >= 2:
                t = pearsonr(stock_datas[x], stock_datas[y])[0]
                return_corr[(x, y)] = t
                corr[(x, y)] = ( t >= FLAG_SIMILARITY or t <= -FLAG_SIMILARITY)
                _t = pearsonr(price_datas[x], price_datas[y])[0]
                price_corr[(x, y)] = ( _t >= FLAG_SIMILARITY or _t <= -FLAG_SIMILARITY)
    
    return corr, price_corr, return_corr

def test(args):
    dataset = Data(args.data_dir, set_name = 'test')
    dataset.prepare_test()
    dataloader = DataLoader(dataset, 1, if_test = True)

    model = SCIModel(args, dataset)
    now_epochs = args.epochs
    print("Now epoch: {}".format(now_epochs))
    pretrain_sd = torch.load('{}/transe_model_sd_epoch{}.ckpt'.format(args.output_dir, now_epochs))
    model.load_state_dict(pretrain_sd,False)
    model.to(args.device)
    model.prepare_test()

    user_ranklist_map = {}
    user_ranklist_score_map = {}
    us_dic = {}
    pred_us_dic = {}

    x = 0
    posts_to_test = dataloader.post_size
    print('size:' , posts_to_test)
    for i in tqdm(range(int(dataloader.user_size / dataloader.batch_size) + 1)):
        if not dataloader.has_next():
            break
        batch_idxs, user_idxs, user_stock, waited_stock_idxs = dataloader.get_test_batch()
        user_stock_scores = model.get_stock_scores(batch_idxs).tolist()
        x+=args.batch_size
        for i in range(len(user_idxs)):
            u_idx  = user_idxs[i]
            us_dic[u_idx] = user_stock[i]

            #sorted_stock_idxs = sorted(range(dataset.stock.size), key=lambda k: user_stock_scores[i][k], reverse=True)
            sorted_stock_idxs = sorted(waited_stock_idxs, key=lambda k: user_stock_scores[i][k], reverse=True)
            pred_us_dic[u_idx] = sorted_stock_idxs 
            user_ranklist_map[u_idx],user_ranklist_score_map[u_idx] = dataset.compute_test_stock_ranklist(u_idx,
                                                user_stock_scores[i], sorted_stock_idxs, args.rank_cutoff) #(stock name, rank)
    
    dataset.output_ranklist(user_ranklist_map, user_ranklist_score_map, args.output_dir)
    evaluate(pred_us_dic, us_dic)
    """
    model.to('cpu')
    all_stock_embed = model.stock(torch.tensor(range(0, dataset.stock.size)))
    model_corr = torch.cosine_similarity(all_stock_embed.unsqueeze(1),all_stock_embed.unsqueeze(0),dim=-1)
    if_rela = (torch.abs(model_corr) >= FLAG_SIMILARITY).tolist()
    model_corr = model_corr.tolist()
    stock2id = {}
    i = 0
    for stock in dataset.stock.vocab:
        stock2id[stock] = i
        i += 1
    corr, price_corr, return_corr = calCorr(args.data_dir, stock2id)

    x, y = 0, 0
    gt,price_gt,pred = [],[],[]
    ground, pred_y = [], []
    for key in tqdm(corr,'comparing'):
        stock_i, stock_j = key
        i, j = stock2id[stock_i], stock2id[stock_j]
        gt.append(corr[key])
        price_gt.append(price_corr[key])
        pred.append(if_rela[i][j])
        if corr[key] == if_rela[i][j]:
            x += 1
        y += 1
        ground.append(return_corr[key])
        pred_y.append(model_corr[i][j])
    print('valid pair: ', y)
    print('For stock return:')
    calculate_metrics(gt, pred)
    #print('For stock price:')
    #calculate_metrics(price_gt, pred)
    X = sm.add_constant(pred_y)
    results = sm.OLS(ground, X).fit()
    print(results.summary())
    """
    print("finish test!!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default = './output/', help='The processed data directory.')
    parser.add_argument('--output_dir', type=str, default='./tmp3/', help='Model output.')
    parser.add_argument('--set_name', type=str, default='./train/', help='Set name: train/test.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--embed_size', type=int, default=300, help='Knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=10, help='Number of negative samples.')
    parser.add_argument('--l2_lambda', type=float, default=0.005, help='L2 lambda')
    parser.add_argument('--hidden_size', type=int, default=100, help='The hidden size of RNN')
    parser.add_argument('--test_mode', type=str, default='test', help='Mode of code: train/test')
    parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--rank_cutoff', type=int, default=100, help='Rank cutoff for output ranklists.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    global logger
    logger = get_logger(args.output_dir + '/train_log.txt')
    logger.info(args)

    if args.test_mode == 'train':
        prepare_dataset(args)
        train(args)
    else:
        test(args)