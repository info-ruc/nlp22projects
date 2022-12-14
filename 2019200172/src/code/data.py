import gzip
import numpy as np
from easydict import EasyDict as edict
import random
import torch
from constant import *
from tqdm import tqdm
import math

class Data:
    def __init__(self, data_dir, set_name='train',word_sampling_rate = 1e-4):
        self.data_dir = data_dir
        self.set_name = set_name 
        self.max_history_length = 20
        self.load_entity()
        self.load_posts()
        self.load_relations()
        self.create_word_sampling_rate(word_sampling_rate)
        #self._clean()
    
    def _load_file(self, filename):
        with gzip.open(self.data_dir+ filename, 'r') as f:
            return [line.decode('utf-8').strip() for line in f]
        
    def load_entity(self):
        # Load Entities from file
        entity_files = edict(
                author='user.txt.gz',
                stock='stock.txt.gz',
                word='vocab.txt.gz',
        )
        self.Gnei = {}
        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            setattr(self, name, edict(vocab=vocab, size=len(vocab))) #
            self.Gnei[name] = [{r:[] for r in get_relations(name)}] * len(vocab)
            print('Load', name, 'of size', len(vocab))
    
    def load_posts(self):
        post_data = []  # (user_idx, stock_idx, [word1_idx,...,wordn_idx])
        word_count = 0
        stock_distrib = np.zeros(self.stock.size)
        word_distrib = np.zeros(self.word.size)
        self.word_count = 0
        for line in self._load_file('{}.txt.gz'.format(self.set_name)):
            arr = line.split('\t')
            if len(arr) <3:
                continue
            user_idx = int(arr[0])
            stock_idx = int(arr[1])
            word_indices = [int(i) for i in arr[2].split(' ')]  # list of word idx
            self.word_count += len(word_indices)
            post_data.append((user_idx, stock_idx, word_indices))
            stock_distrib[stock_idx] += 1
            for wi in word_indices:
                word_distrib[wi] += 1
            word_count += len(word_indices)

        self.post = edict(
                data=post_data,
                size=len(post_data),
                stock_distrib=stock_distrib,
                stock_uniform_distrib=np.ones(self.stock.size),
                word_distrib=word_distrib,
                word_count=word_count,
                post_distrib=np.ones(len(post_data)) #set to 1 now
        )
        print('Load posts of size', self.post.size, 'word count=', word_count)
    
    def _add_edge(self, eid1, etype1, eid2, etype2, relation):
        self.Gnei[etype1][eid1][relation].append(eid2)
        self.Gnei[etype2][eid2][relation].append(eid1)
	      
    def load_relations(self):
        self.user_stock = {i: set() for i in range(self.author.size)}
        n_edge = 0
        for user_idx, stock_idx, words_list in tqdm(self.post.data, 'build graph'):
            self.user_stock[user_idx].add(stock_idx)
            self._add_edge(user_idx, AUTHOR, stock_idx, STOCK, FOLLOW)
            n_edge += 2
            for w in words_list:
                self._add_edge(user_idx, AUTHOR, w, WORD, MENTION)
                self._add_edge(stock_idx, STOCK, w, WORD, DESCRIBED_AS)
                n_edge += 4
        print('Total edge size: ,', n_edge)

    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.post.word_distrib) * sampling_threshold
        for i in range(self.word.size):
            if self.post.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min((np.sqrt(float(self.post.word_distrib[i]) / threshold) + 1) * threshold / float(self.post.word_distrib[i]), 1.0)
        
    def _clean(self):
        for etype in self.Gnei:
            for eid in tqdm(range(len(self.Gnei[etype])), 'Remove duplicates for {}'.format(etype)):
                for r in self.Gnei[etype][eid]:
                    data = self.Gnei[etype][eid][r]
                    self.Gnei[etype][eid][r] = list(sorted(set(data)))
    
    def prepare_test(self):
        self.user_idxs = range(self.author.size)
        self.user_train_stock_set_list = {i:set() for i in range(self.author.size)}
        for line in self._load_file('{}.txt.gz'.format(self.set_name)):
            arr = line.split('\t')
            if len(arr) <3:
                continue
            user_idx = int(arr[0])
            stock_idx = int(arr[1])
            self.user_train_stock_set_list[user_idx].add(stock_idx)
        
        self.user_test_stock_set_list = {i:set() for i in range(self.author.size)}
        for u,s,_ in self.post.data:
            self.user_test_stock_set_list[u].add(s)

    def compute_test_stock_ranklist(self, u_idx, original_scores, sorted_stock_idxs, rank_cutoff):
        stock_rank_list = []
        stock_rank_scores = []
        rank = 0
        for stock_idx in sorted_stock_idxs:
            if stock_idx in self.user_train_stock_set_list[u_idx] or math.isnan(original_scores[stock_idx]):
                continue
            stock_rank_list.append(stock_idx)
            stock_rank_scores.append(original_scores[stock_idx])
            rank += 1
            if rank == rank_cutoff:
                break
        return stock_rank_list, stock_rank_scores

    def output_ranklist(self, user_ranklist_map, user_ranklist_score_map, output_dir):
        with open(output_dir + 'test.ranklist', 'w') as rank_fout:
            for u in user_ranklist_map:
                user_id = self.author.vocab[u]
                rank = 1
                for i in range(len(user_ranklist_map[u])):
                    if user_ranklist_map[u][i] < 0 or user_ranklist_map[u][i] >= self.stock.size :
                        continue
                    stock_id = self.stock.vocab[user_ranklist_map[u][i]]
                    rank_fout.write(user_id+' Q0 ' + stock_id + ' ' + str(rank)
                            + ' ' + str(user_ranklist_score_map[u][i]) + ' StockCorrelationIdentification\n')
                    rank += 1

class DataLoader(object):
    def __init__(self, dataset, batch_size, rank_cut = 100, if_test = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.post_size = self.dataset.post.size
        if if_test:
            self.user_size = self.dataset.author.size
        self.finished_word_num = 0
        self.finished_post_num = 0
        self.rank_cut = max(rank_cut, batch_size)
        self.reset()
    
    def reset(self):
        # Shuffle posts order
        self.post_seq = np.random.permutation(self.post_size)
        self.cur_post_i = 0
        self.cur_word_i = 0
        self._has_next = True
        self.all_stocks = set(range(self.dataset.stock.size))

    def get_batch(self):
        user_idxs, stock_idxs, neg_stock_idxs, word_idxs = [],[],[],[]
        post_idx = self.post_seq[self.cur_post_i]
        user_idx, stock_idx, text_list = self.dataset.post.data[post_idx]

        while len(user_idxs) < self.batch_size:
            word_idx = text_list[self.cur_word_i]
            if random.random() < self.dataset.word_sampling_rate[word_idx]:
                user_idxs.append(user_idx)
                stock_idxs.append(stock_idx)
                word_idxs.append(word_idx)
                
            self.cur_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i >= len(text_list):
                self.cur_post_i += 1
                if self.cur_post_i >= self.post_size:
                    self._has_next = False
                    break
                self.cur_word_i = 0
                post_idx = self.post_seq[self.cur_post_i]
                user_idx, stock_idx, text_list = self.dataset.post.data[post_idx]

        return [torch.tensor(user_idxs), torch.tensor(stock_idxs), torch.tensor(word_idxs)]
    
    def get_test_batch(self):
        user_idxs, stock_idxs, waited_stock_idxs = [],[],[]
        # Start from a new user
        min_len = min(self.batch_size, self.user_size-self.cur_post_i)
        for i in range(min_len):
            u = self.dataset.user_idxs[self.cur_post_i]
            user_idxs.append(u)
            stock_idxs.append(self.dataset.user_test_stock_set_list[u])
            waited_stock_idxs.extend(stock_idxs[-1])
            self.cur_post_i += 1
            self.finished_post_num += 1
        
        waited_stock_idxs = set(waited_stock_idxs)
        s_len = max(0, self.rank_cut - len(waited_stock_idxs))
        other_stock_idxs = np.random.choice(list(self.all_stocks - waited_stock_idxs), s_len, replace = False)
        for s in other_stock_idxs:
            waited_stock_idxs.add(s)

        if self.cur_post_i >= self.user_size:
            self._has_next = False
        return [torch.tensor(user_idxs), torch.tensor(list(waited_stock_idxs))], user_idxs, stock_idxs, waited_stock_idxs


    def has_next(self):
        """Has next batch."""
        return self._has_next
