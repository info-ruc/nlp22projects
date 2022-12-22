#! /home/user1/venv/vpy38/bin/python
# _*_ coding:UTF-8 _*_


import pandas as pd
import torch
import random
from torch import nn
from transformers import BertModel,BertTokenizer

from tqdm import tqdm




# read data
sample = pd.read_pickle('tagged_sample.pkl')
sample = sample.sample(frac=1)
sample.index = range(len(sample))
sample['contents'] = sample['contents'].apply(lambda x:x[:510])
m = {-1:0,1:1}
sample['flag'] = sample['flag'].apply(lambda x:m[x])


# split the training set

validate_rate = 0.4
len_train = int(len(sample) * (1-validate_rate))
print('size of training set : ',len_train)

train_contents, train_labels = list(sample[: len_train]['contents']), list(sample[: len_train]['flag'])
test_contents, test_labels = list(sample[len_train:]['contents']), list(sample[len_train:]['flag'])



class BERTClassifier(nn.Module):

    # 此处使用熵简科技利用中文金融数据训练的 FinBERT 模型
    def __init__(self, output_dim, pretrained_name='FinBERT/'):

        super(BERTClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.mlp = nn.Linear(768, output_dim)


    def forward(self, tokens_X):

        res = self.bert(**tokens_X)
        return self.mlp(res[1])



def evaluate(net, comments_data, labels_data):
    
    sum_correct, i = 0, 0
    
    for i in (range(int(len(comments_data)//8))):
        i = i*8
        comments = comments_data[i: min(i + 8, len(comments_data))]
        tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt',max_length=512).to(device=device)
        res = net(tokens_X)                                          # 获得到预测结果
        y = torch.tensor(labels_data[i: min(i + 8, len(comments_data))]).reshape(-1).to(device=device)

        sum_correct += (res.argmax(axis=1) == y).sum()              # 累加预测正确的结果


    return sum_correct/len(comments_data)                           # 返回(总正确结果/所有样本)，精确率



def train_bert_classifier(net, tokenizer, loss, optimizer, train_comments, train_labels, test_comments, test_labels,
                          device, epochs):
    max_acc = 0.5  # 初始化模型最大精度为0.5

    
    # 先测试未训练前的模型精确度
    train_acc = evaluate(net, train_comments, train_labels)
    test_acc = evaluate(net, test_comments, test_labels)
    # 输出精度
    print('--epoch', 0, '\t--train_acc:', train_acc, '\t--test_acc', test_acc)


    # 累计训练18万条数据 epochs 次，优化模型
    for epoch in tqdm(range(epochs)):

        i, sum_loss = 0, 0  # 每次开始训练时， i 为 0 表示从第一条数据开始训练


        # 开始训练模型
        while i < len(train_comments):
            comments = train_comments[i: min(i + 8, len(train_comments))]  # 批量训练，每次训练8条样本数据

            # 通过 tokenizer 数据化输入的评论语句信息，准备输入bert分类器
            tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)

            # 将数据输入到bert分类器模型中，获得结果
            res = net(tokens_X)

            # 批量获取实际结果信息
            y = torch.tensor(train_labels[i: min(i + 8, len(train_comments))]).reshape(-1).to(device=device)

            optimizer.zero_grad()  # 清空梯度
            l = loss(res, y)  # 计算损失
            l.backward()  # 后向传播
            optimizer.step()  # 更新梯度

            sum_loss += l.detach()  # 累加损失
            i += 8  # 样本下标累加


        # 计算训练集与测试集的精度
        train_acc = evaluate(net, train_comments, train_labels)
        test_acc = evaluate(net, test_comments, test_labels)

        # 输出精度
        print('\n--epoch', epoch+1, '\t--loss:', sum_loss / (len(train_comments) / 8), '\t--train_acc:', train_acc,
              '\t--test_acc', test_acc)

        # 如果测试集精度 大于 之前保存的最大精度，保存模型参数，并重设最大值
        if test_acc > max_acc:
            # 更新历史最大精确度
            max_acc = test_acc

            # 保存模型
            torch.save(net.state_dict(), 'bert.parameters')



if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = BERTClassifier(output_dim=2)                      # BERTClassifier分类器
    net = net.to(device)                                    

    tokenizer = BertTokenizer.from_pretrained("FinBERT/")
    #bert = BertModel.from_pretrained("FinBERT/")
    loss = nn.CrossEntropyLoss()                                # 损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)      # 小批量随机梯度下降算法
    train_bert_classifier(net, tokenizer, loss, optimizer, train_contents, train_labels, test_contents, test_labels, device, 30)

