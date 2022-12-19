from bert_seq2seq.tokenizer import Tokenizer
from bert_seq2seq import Predictor
from bert_seq2seq import load_model
import torch
import random
import pandas as pd
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu
model_name = "roberta"  # 选择模型名字
task_name = "seq2seq"  # 任务名字
model_path = "./roberta_auto_title_model.bin"
vocab_path = "./state_dict/roberta/vocab.txt"
data_dir='./data/lcsts_data.json'
tokenizer = Tokenizer(vocab_path)
bert_model = load_model(tokenizer.vocab, model_name=model_name, task_name=task_name)
bert_model.load_all_params(model_path)
predictor = Predictor(bert_model, tokenizer)

if __name__ == '__main__':
    df=pd.read_json(data_dir)
    #l=random.randint(100001,500000)
    l=100001
    textset=list(df['content'][l:l+100])
    i=0
    references=[] #真实值
    hypothesis=[] #预测值
    for text in textset:
        out=predictor.predict_generate_beamsearch(text,beam_size=3,input_max_length=200,out_max_length=40)
        print(out)
        print(df['title'][l+i])
        reference=df['title'][l+i]
        hypo=' '.join(jieba.cut(out))
        reference=' '.join(jieba.cut(reference))
        references.append(reference)
        hypothesis.append(hypo)
        i=i+1
    rouge=Rouge()
    scores = rouge.get_scores(references, hypothesis,avg=True)
    print(scores)