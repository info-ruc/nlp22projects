import pickle
import logging.handlers
import logging
import sys
from math import log
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score

AUTHOR = 'author'
STOCK = 'stock'
POST = 'post'
WORD = 'word'

POST_TYPE = 'post type'
TITLE = 'title'
LIKE_COUNT = 'like count'
CONTENTS = 'contents'
URL = 'url'
VIEW_COUNT = 'view count'
REPONSE_COUNT = 'reponse count'
TIME = 'time'
CUT_WORDS = 'cut_words'

MENTION = 'mention'
FOLLOW = 'follow'
DESCRIBED_AS = 'described_as'

KG_RELATION = {
    AUTHOR: {
        MENTION: WORD,
        FOLLOW: STOCK,
    },
    WORD: {
        MENTION: AUTHOR,
        DESCRIBED_AS: STOCK,
    },
    STOCK: {
        FOLLOW: AUTHOR,
        DESCRIBED_AS: WORD, 
    }
}

def get_relations(entity_type):
    return list(KG_RELATION[entity_type].keys())

def save_dataset(output_dir, dataset_obj):
    dataset_file = output_dir + '/dataset.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)

def load_dataset(output_dir):
    dataset_file = output_dir + '/dataset.pkl'
    print('Load dataset from: ', output_dir)
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj

def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def evaluate(topk_matches, test_user_stocks, top=100):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of stock ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    y=0;z=0;u=0;w=0
    precisions, recalls, ndcgs, ndcgs10, hits, mrr, map_ = [], [], [], [], [], [], []
    test_user_idxs = list(test_user_stocks.keys())
    # uids2 = list(topk_matches.keys())
    # print(test_user_idxs)
    # print(topk_matches)
    # print(set(uids2)&set(test_user_idxs))
    print(len(test_user_idxs))
    for uid in tqdm(test_user_idxs):
        # print(uid)
        u+=1
        # print(len(test_user_query_stocks[uid]))
        if uid not in topk_matches:
            w+=1
            # print(uid)
            # print(uid in train)
        if uid not in topk_matches or len(topk_matches[uid]) < top:
            invalid_users.append(uid)
            # print(uid,qid)
            # print(len(topk_matches[uid][qid]))
            y+=1
            continue
        pred_list, rel_set = topk_matches[uid], test_user_stocks[uid]
        # print('len:',len(pred_list),len(rel_set))
        if len(rel_set) == 0:
            z+=1
            continue
            # print("?")
        dcg = 0.0
        dcg10 = 0.0
        hit_num = 0.0
        mrr_num = 0.0
        map_num = 0.0

        for i, values in enumerate(pred_list):
            if values in rel_set:
                if i < 10:
                    dcg10 += 1. / (log(i + 2) / log(2))
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
                mrr_num += 1.0 / (i + 1.0)
                map_num += hit_num / (i + 1.0)
        # idcg
        idcg, idcg10 = 0.0, 0.0
        for i in range(min(len(rel_set), 10)):
            idcg10 += 1. / (log(i + 2) / log(2))

        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg
        ndcg10 = dcg10 / idcg10
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0
        map_num = map_num / max(1.0, len(rel_set)) 

        ndcgs.append(ndcg)
        ndcgs10.append(ndcg10)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)
        mrr.append(mrr_num)
        map_.append(map_num)
        # print(ndcg,recall,precision,hit,hit_num)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    #avg_ndcgs10 = np.mean(ndcgs10) * 100
    avg_hit = np.mean(hits) * 100
    avg_mrr = np.mean(mrr) * 100
    avg_map = np.mean(map_) * 100
    print(u,w,y,z)

    print('MAP={:.3f} |  MRR={:.3f} | NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
            avg_map, avg_mrr, avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))

def calculate_metrics(gt, pred):
    print("starting!!!-----------------------------------------------")
    # 打印具体的混淆矩阵的每个部分的值
    confusion = confusion_matrix(gt, pred)
    print(confusion)
    # 从左到右依次表示TN、FP、FN、TP
    print(confusion.ravel())
    # 绘制混淆矩阵的图
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    # 通过混淆矩阵计算每个评估指标的值
    print('AUC:',roc_auc_score(gt, pred))
    print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
    print('Sensitivity:', TP / float(TP + FN))
    print('Recall:',TP / float(TP + FN))
    print('Precision:',TP / float(TP + FP))
    # 用于计算F1-score = 2*recall*precision/recall+precision,这个情况是比较多的
    P = TP / float(TP + FP)
    R = TP / float(TP + FN)
    print('F1-score:',(2*P*R)/(P+R))
    print('True Positive Rate:',round(TP / float(TP + FN)))
    print('False Positive Rate:',FP / float(FP + TN))
    print('Ending!!!------------------------------------------------------')
 
    # 采用sklearn提供的函数验证,用于对比混淆矩阵方法与这个方法的区别
    print("the result of sklearn package")
    auc = roc_auc_score(gt,pred)
    print("sklearn auc:",auc)
    accuracy = accuracy_score(gt,pred)
    print("sklearn accuracy:",accuracy)
    recal = recall_score(gt,pred)
    precision = precision_score(gt,pred)
    print("sklearn recall:{},precision:{}".format(recal,precision))
    print("sklearn F1-score:{}".format((2*recal*precision)/(recal+precision)))
