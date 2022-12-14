from extract import *
from constant import *
import argparse
import jieba
import gzip
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2017, help='the year you want to extract data from.')
parser.add_argument('--stop_file', type=str, default='./data/hit_stopwords.txt', help='the stopwords file.')
parser.add_argument('--output_dir', type=str, default='./output_0.3/', help='output directory.')
parser.add_argument('--split_rate', type=float, default=0.3, help='the stopwords file.')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

with open(args.stop_file, 'r', encoding='utf-8') as f:    # 
    stopword_list = [word.strip('\n') for word in f.readlines()]

data = read_data(args.year)
u_s_dict = {}
for url in tqdm(data):
    uid = data[url][AUTHOR]
    stock_id = data[url][STOCK]
    if uid not in u_s_dict:
        u_s_dict[uid] = set()
    u_s_dict[uid].add(stock_id)
# 删除只follow了一个stock的user
removed_user = [u for u in u_s_dict if len(u_s_dict[u])<=1]

def clear_and_cut_text(text):
    images = IMG_PATTERN.findall(text)
    text = SUB_PATTERN.sub('', text)
    text = re.sub('[\n\r ]', '', text)
    images.append(text)

    word_list = jieba.lcut(' '.join(images))
    result = []
    for w in word_list:
        if w not in stopword_list:
            result.append(w)
    
    return result

# index the user ID and stock ID
authors, stocks = [], []
authors_dict, stocks_dict = {}, {}
words = set()
result_data = []
user_post_map = {}

i = 0
u_s_dict = {}
for url in tqdm(data,'index and clear data'):
    post = data[url]
    uid = post[AUTHOR]
    stock_id = post[STOCK]

    if uid in removed_user:
        continue
    if uid not in authors:
        authors_dict[uid] = str(len(authors))
        authors.append(uid)
        user_post_map[uid] = []
    if stock_id not in stocks:
        stocks_dict[stock_id] = str(len(stocks))
        stocks.append(stock_id)
    cut_list = clear_and_cut_text(post[TITLE] + ' ' + post[CONTENTS])
    for w in cut_list:
        words.add(w)
    
    post['cut_words'] = cut_list
    user_post_map[uid].append(i)
    result_data.append(post)
    i += 1

    if uid not in u_s_dict:
        u_s_dict[uid] = set()
    u_s_dict[uid].add(stock_id)

print('user size: {}; stock size: {}; words size: {}'.format(len(authors), len(stocks), len(words)))
print('remained posts account for: {:.2f}%'.format(len(result_data)/len(data)*100))

with gzip.open(args.output_dir + 'user.txt.gz','wt') as fout:
	for u in authors:
		fout.write(u + '\n')

with gzip.open(args.output_dir + 'stock.txt.gz','wt') as fout:
	for s in stocks:
		fout.write(s + '\n')

words = list(words)
with gzip.open(args.output_dir + 'vocab.txt.gz','wt') as fout:
	for w in words:
		fout.write(w + '\n')

test_u_s = [] # test user-stock pairs
sum_size = 0
for u in u_s_dict:
    sample_number = int(len(u_s_dict[u]) * args.split_rate)
    sum_size += len(u_s_dict[u])
    t_stocks = np.random.choice(list(u_s_dict[u]), sample_number , replace=False)
    for s in t_stocks:
        test_u_s.append((u, s))

words_dict= {}
for i in range(len(words)):
    words_dict[words[i]] = str(i)
    i += 1

test_i, train_i = 0, 0
with gzip.open(args.output_dir + 'train.txt.gz', 'wt') as train_fout, gzip.open(args.output_dir + 'test.txt.gz', 'wt') as test_fout:
    for post in result_data:
        if len(post[CUT_WORDS]) == 0:
            continue
        user_id = post[AUTHOR]
        stock_id = post[STOCK]
        words_text = ' '.join([words_dict[w] for w in post[CUT_WORDS]])
        if (user_id, stock_id) in test_u_s:
            test_fout.write(authors_dict[user_id] + '\t' + stocks_dict[stock_id] + '\t' + words_text.strip() + '\n')
            test_i += 1
        else:
            train_fout.write(authors_dict[user_id] + '\t' + stocks_dict[stock_id] + '\t' + words_text.strip() + '\n')
            train_i += 1

print('POST: test size {}, train size {}, test accounts for {:.2f}%;'.format(test_i, train_i, test_i / (test_i+train_i) * 100))
print('US_Pair: test size {}, test accounts for {:.2f}%.'.format(len(test_u_s), len(test_u_s) / sum_size * 100))