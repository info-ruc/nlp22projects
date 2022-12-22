import torch 
import torch.nn as nn
from easydict import EasyDict as edict
import numpy as np
from constant import *
import torch.nn.functional as F

class SCIModel(nn.Module):
    def __init__(self, args, dataset):
        super(SCIModel, self).__init__()
        self.dataset = dataset
        self.embed_size = args.embed_size
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        self.hidden_size = args.hidden_size
        self.num_neg_samples = args.num_neg_samples

        def _entity_embedding(vocab_size):
            embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
            #initrange = 0.5 #/ self.embed_size
            #weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
            #embed.weight = nn.Parameter(weight)
            nn.init.xavier_uniform_(embed.weight)
            return embed

        def _make_distrib(distrib):
            distrib = np.power(np.array(distrib, dtype=np.float), 0.75)
            distrib = distrib / distrib.sum()
            distrib = torch.FloatTensor(distrib).to(self.device)
            return distrib
        
        self.entities = edict(
            author=edict(size=dataset.author.size),
            stock=edict(size=dataset.stock.size),
            word=edict(size=dataset.word.size),
        )
        for e in self.entities:
            embed = _entity_embedding(getattr(self.dataset, e).size)
            setattr(self, e, embed)

        self.relations = edict(
            follow=edict(
                et='stock',
                et_distrib=_make_distrib(dataset.post.stock_uniform_distrib)),
            mention=edict(
                et='word',
                et_distrib=_make_distrib(dataset.post.word_distrib)),
            described_as=edict(
                et='word',
                et_distrib=_make_distrib(dataset.post.word_distrib)),
        )

        def _relation_embedding():
            """Create relation vector of size [1, embed_size]."""
            initrange = 0.5 / self.embed_size
            weight = torch.FloatTensor(1, self.embed_size)#.uniform_(-initrange, initrange)
            embed = nn.Parameter(weight)
            nn.init.xavier_uniform_(embed)
            return embed

        def _relation_bias(size):
            """Create relation bias of size [vocab_size+1]."""
            bias = nn.Embedding(size + 1, 1, padding_idx=-1, sparse=False)
            bias.weight = nn.Parameter(torch.zeros(size + 1, 1))
            return bias

        for r in self.relations:
            embed = _relation_embedding()
            setattr(self, r, embed)
            bias = _relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def forward(self, batch_idxs):
        user_idxs = batch_idxs[0].to(self.device)
        stock_idxs = batch_idxs[1].to(self.device)
        word_idxs = batch_idxs[2].to(self.device)

        regularizations = []

        us_loss, us_embeds = self.neg_loss(AUTHOR, FOLLOW, STOCK, user_idxs, stock_idxs)
        regularizations.extend(us_embeds)
        loss = us_loss

        uw_loss, uw_embeds = self.neg_loss(AUTHOR, MENTION, WORD, user_idxs, word_idxs)
        regularizations.extend(uw_embeds)
        loss += uw_loss

        sw_loss, sw_embeds = self.neg_loss(STOCK, DESCRIBED_AS, WORD, stock_idxs, word_idxs)
        regularizations.extend(sw_embeds)
        loss += sw_loss

        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss
        
        return loss
    
    def prepare_test(self):
        all_stock_idxs = torch.tensor(range(0, self.dataset.stock.size)).to(self.device)
        stock_embed = self.stock(all_stock_idxs)
        self.stock_vec = torch.transpose(stock_embed, 0, 1)
        self.relation_bias = self.follow_bias(all_stock_idxs).squeeze(1)

    def get_stock_scores(self, batch_idxs):
        user_idxs = batch_idxs[0].to(self.device)
        waited_idxs = batch_idxs[1].to(self.device)
        user_embed = self.author(user_idxs)
        relation_vec = self.follow
        example_vec = user_embed + relation_vec

        #stock_vec  = self.stock(waited_idxs)
        #relation_bias = self.follow_bias(waited_idxs).squeeze(1)
        #scores = torch.matmul(example_vec, torch.transpose(stock_vec, 0, 1)) + relation_bias
        
        scores = torch.matmul(example_vec, self.stock_vec) + self.relation_bias
        return scores

    def neg_loss(self, etype_head, relation, etype_tail, head_idxs, tail_idxs):
        mask = tail_idxs >= 0
        fixed_entity_head_idxs = head_idxs[mask]
        fixed_entity_tail_idxs = tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, [] 
        
        head_embedding = getattr(self, etype_head)
        tail_embedding = getattr(self, etype_tail)
        relation_vec = getattr(self, relation)
        relation_bias_embedding = getattr(self, relation + '_bias')
        tail_distrib = self.relations[relation].et_distrib

        return kg_neg_loss(head_embedding, tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, relation_bias_embedding, self.num_neg_samples, tail_distrib)


def kg_neg_loss(head_embed, tail_embed, head_idxs, tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib):
    
    head_vec = head_embed(head_idxs)  # [batch_size, embed_size]
    example_vec = head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

    tail_vec = tail_embed(tail_idxs)  # [batch_size, embed_size]
    relation_bias = relation_bias_embed(tail_idxs).squeeze(1)  # [batch_size]
    pos_vec = tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean() # nce_loss(pos_logits, neg_logits) # 
    return loss, [head_vec, tail_vec, neg_vec]

def nce_loss(true_logits, sampled_logits):
    true_xent = F.binary_cross_entropy_with_logits(F.sigmoid(true_logits), torch.ones_like(true_logits), reduction='none')   # 测量离散分类任务中的概率误差				
    sampled_xent = F.binary_cross_entropy_with_logits(F.sigmoid(sampled_logits), torch.zeros_like(sampled_logits), reduction='none')   # 测量离散分类任务中的概率误差							
    
    nce_loss_tensor = (true_xent +sampled_xent.mean(1))
    return nce_loss_tensor.mean()