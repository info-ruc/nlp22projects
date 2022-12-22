import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
import stanza
from nltk.tokenize import RegexpTokenizer
import torch
from torch import nn, optim
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset
from dgl.nn.pytorch import GraphConv
import networkx as nx 
from scipy.sparse import csr_matrix
from nltk.tokenize import RegexpTokenizer

from time import sleep
from tqdm import tqdm

def Get_train_dataset(path="./data/msr_paraphrase_train.txt"):
    print("\n")
    print("start: get train dataset.\n")
    print("waiting......\n")
    f =open(path, encoding='utf-8')
    Quality = []
    Senten_1 = []
    Senten_2 = []

    flag = 0
    for line in f.readlines():
        flag = flag + 1
        if flag == 1:
            continue
        line = line.strip('\n')
        line = line.split('\t')
        Quality.append(line[0])
        Senten_1.append(line[3])
        Senten_2.append(line[4])
 
    datas = {'senten1':Senten_1, 'senten2':Senten_2, 'quality':Quality}
    dataset_df = pd.DataFrame(datas)
    f.close()

    #dataset_df = dataset_df[:20]
    print("success: get train dataset.\n")

    return dataset_df

def Get_test_dataset(path='./data/msr_paraphrase_test.txt'):
    print("\n")
    print("start: get test dataset.\n")
    print("waiting......\n")
    f =open(path, encoding='utf-8')
    Quality_test = []
    Senten_1_test = []
    Senten_2_test = []

    flag = 0
    for line in f.readlines():
        flag = flag + 1
        if flag == 1:
            continue
        line = line.strip('\n')
        line = line.split('\t')
        Quality_test.append(line[0])
        Senten_1_test.append(line[3])
        Senten_2_test.append(line[4])


    datas_test = {'senten1':Senten_1_test, 'senten2':Senten_2_test, 'quality':Quality_test}
    dataset_df_test = pd.DataFrame(datas_test)
    f.close()

    #dataset_df_test = dataset_df_test[:10]

    print("success: get test dataset.\n")
    return dataset_df_test

#Get the word embeddings
def Get_wordembedding(path="./data/glove.42B.300d.txt"):
    embeddings_dict = {}

    print("start: get word embeddings.\n")
    print("waiting......\n")
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector


    f.close()
    print("success: get word embeddings.\n")
    return embeddings_dict

def Sentence_to_matrix(df, embeddings_dict):

    # Download the language model
    # stanza.download('en',download_method=None)
    # Build a Neural Pipeline
    nlp = stanza.Pipeline('en', r".\stanza_resources", processors = "tokenize,mwt,pos,lemma,depparse", download_method=None)
    
    all_syntax_matrix_1 = np.zeros((len(df), 35, 35))
    all_syntax_matrix_2 = np.zeros((len(df), 35, 35))
    
    all_sentence_matrix_1 = []
    all_sentence_matrix_2 = []
    
    print("\n")
    print("start: get sentence matrix.\n")
    print("waiting......(about 12 minute for train data, 6 minute for test data)")

    for i in range(len(df)):
        # sentences 
        s_1 = df.iloc[i, 0]
        s_2 = df.iloc[i, 1]
    
        #get tokens from sentence
        doc_1 = nlp(s_1)
        sent_dict_1 = doc_1.sentences[0].to_dict()
        doc_2 = nlp(s_2)
        sent_dict_2 = doc_2.sentences[0].to_dict()
        
        
        origtoken_1 = []
        curtoken_1 = []
        punc_loc_1 = []
        sen_emd_1 = []
        
        origtoken_2 = []
        curtoken_2 = []
        punc_loc_2 = []
        sen_emd_2 = []
        
        # construct origin syntax relation
        size_1 = len(sent_dict_1)
        origsyntax_1 = np.zeros((size_1, size_1)).astype(int)
        
        size_2 = len(sent_dict_2)
        origsyntax_2 = np.zeros((size_2, size_2)).astype(int)
        
        # ------------------------------------------------sentence 1---------------------------------------------------
        for word in sent_dict_1:
            # get original word list
            origtoken_1.append(word['text'])
            
            # get punctuation location \ word list without punctuation \ sentence embedding
            if word['text'] in [',','','.','-',"'",'?','"']:
                punc_loc_1.append(word['id']-1)
            else:
                curtoken_1.append(word['text'])
                if word['text'].lower() in embeddings_dict:
                    sen_emd_1.append(np.array(embeddings_dict[word['text'].lower()]))
                else:
                    sen_emd_1.append((np.random.rand(300) / 5 - 0.1))
                # also can test oov word embedding in range of(-0.1, 0.1)
        
            # get word's relation
            if word['head'] == 0:
                continue
            else:
                origsyntax_1[word['head']-1, word['id']-1] = 1
        
        #get fixed length of sentence embedding
        while len(sen_emd_1) < 35:
                sen_emd_1.append(np.array([0] * 300))
            
            
        
        # delete relation of punctuation
        syntax_1 = np.delete(origsyntax_1,punc_loc_1, axis = 0)
        syntax_1 = np.delete(syntax_1,punc_loc_1, axis = 1)
        
        # put current sentence embedding into matrix
        all_sentence_matrix_1.append(np.array(sen_emd_1))
        
        # put current syntax matrix into matrix
        all_syntax_matrix_1[i, :syntax_1.shape[0], :syntax_1.shape[1]] = syntax_1.copy()
        
        
        # ------------------------------------------------sentence 2---------------------------------------------------
        for word in sent_dict_2:
            # get original word list
            origtoken_2.append(word['text'])
            
            # get punctuation location \ word list without punctuation \ sentence embedding
            if word['text'] in [',','','.','-',"'",'?','"']:
                punc_loc_2.append(word['id']-1)
            else:
                curtoken_2.append(word['text'])
                if word['text'].lower() in embeddings_dict:
                    sen_emd_2.append(np.array(embeddings_dict[word['text'].lower()]))
                else:
                    sen_emd_2.append((np.random.rand(300) / 5 - 0.1))
                # also can test oov word embedding in range of(-0.1, 0.1)
        
            # get word's relation
            if word['head'] == 0:
                continue
            else:
                origsyntax_2[word['head']-1, word['id']-1] = 1
        
        #get fixed length of sentence embedding
        while len(sen_emd_2) < 35:
                sen_emd_2.append(np.array([0] * 300))
            
            
        
        # delete relation of punctuation
        syntax_2 = np.delete(origsyntax_2,punc_loc_2, axis = 0)
        syntax_2 = np.delete(syntax_2,punc_loc_2, axis = 1)
        
        # put current sentence embedding into matrix
        all_sentence_matrix_2.append(np.array(sen_emd_2))
        
        # put current syntax matrix into matrix
        all_syntax_matrix_2[i, :syntax_2.shape[0], :syntax_2.shape[1]] = syntax_2.copy()

    print("success: get sentence matrix.\n")

    return np.array(all_sentence_matrix_1), np.array(all_sentence_matrix_2), all_syntax_matrix_1, all_syntax_matrix_2

def Matrix_to_graph(senten_1,senten_2,syntax_1,syntax_2):
    graph_1 = []
    graph_2 = []
    
    senten_1 = torch.FloatTensor(senten_1)
    senten_2 = torch.FloatTensor(senten_2)

    print("\n")
    print("start: get sentence graph.\n")
    print("waiting......")

    
    for i in range(senten_1.shape[0]):
        # create one graph for one sentence
        # without direction
        # g_1 = dgl.from_networkx(nx.from_numpy_matrix(syntax_1[i, :, :]))
        # g_2 = dgl.from_networkx(nx.from_numpy_matrix(syntax_2[i, :, :]))
        
        # with ditrection
        g_1 = dgl.from_scipy(csr_matrix(syntax_1[i, :, :]))
        g_2 = dgl.from_scipy(csr_matrix(syntax_2[i, :, :]))
        
        # set words feature for one sentence
        g_1.ndata['x'] = senten_1[i,:,:]
        g_2.ndata['x'] = senten_2[i,:,:]

        g_1 = dgl.add_self_loop(g_1)
        g_2 = dgl.add_self_loop(g_2)
        
        # add
        graph_1.append(g_1)
        graph_2.append(g_2)
    
    print("success: get sentence graph.\n")
    
    return graph_1, graph_2


# change sentence to matrix
def Sentence_to_matrix_lstm(df, embeddings_dict):
    tokenizer = RegexpTokenizer(r'\w+')
    print("Build sentence matrix.")
    sentence_matrix_1 = []
    sentence_matrix_2 = []
    for i in tqdm(range(len(df))):
        sleep(0.1)
        sentence_1 = tokenizer.tokenize(df.iloc[i, 0].lower())
        sentence_2 = tokenizer.tokenize(df.iloc[i, 1].lower())
        s1 = []
        s2 = []
        
        for word in sentence_1:
            if word in embeddings_dict:
                s1.append(np.array(embeddings_dict[word]))
            else:
                s1.append((np.random.rand(300) / 5 - 0.1))
            # also can test oov word embedding in range of(-0.1, 0.1)
        while len(s1) < 35:
            s1.append(np.array([0] * 300))
        sentence_matrix_1.append(np.array(s1))
        
        for word in sentence_2:
            if word in embeddings_dict:
                s2.append(np.array(embeddings_dict[word]))
            else:
                s2.append((np.random.rand(300) / 5 - 0.1))
            # also can test oov word embedding in range of(-0.1, 0.1)
        while len(s2) < 35:
            s2.append(np.array([0] * 300))
        sentence_matrix_2.append(np.array(s2))
    
    print("Finish building sentence matrix.")
    return np.array(sentence_matrix_1), np.array(sentence_matrix_2)

def Get_score_gcn(model, df_dataset_test, nums_graph, g1_test, g2_test, labels_test):
    
    result = np.zeros((0, 2))

    for i in range(0, int(len(df_dataset_test)/nums_graph)):
        las = nums_graph * i
        #g1, g2, labels_test = samples
        g_1_test = dgl.batch(g1_test[las:las+nums_graph])
        g_2_test = dgl.batch(g2_test[las:las+nums_graph])
        
            
        pred = model(g_1_test, g_2_test)
        p = pred.detach().numpy()
        result = np.concatenate((result, p))
    
    result = np.array(result)

    y_true = []
    y_pred = []

    for i in range(int(len(labels_test)/nums_graph)*nums_graph):
        if labels_test[i, 0] == 1:
            y_true.append(0)
        else:
            y_true.append(1)
            
        if result[i, 0] > result[i, 1]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return f1, acc