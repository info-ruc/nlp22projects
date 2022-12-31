# %%
# import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import math
import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel #, get_linear_schedule_with_warmup
# from tqdm import tqdm, trange
#import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.lm import Vocabulary
from utils import *

#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

# %%
dataset_json = json.load(open("./reddit_chat.json"))
dataset = []

for item in dataset_json.values():
    for i in item['content']:
        dataset.append(i['message'])

train_dataset = dataset[:-100]
test_dataset = dataset[-100:]

# %%
train_dataset = Entries(train_dataset, tokenizer, truncate=False)

test_dataset_true_end = [' '.join(x.split()[math.floor(len(x.split())*0.8):]) for x in test_dataset]
test_dataset_start = [' '.join(x.split()[:math.floor(len(x.split())*0.8)]) for x in test_dataset]

# #%%
# generate(model.to('cuda'), tokenizer,"Jeff Bezos is evil because", entry_count=50)
# #%%
# generated = text_generation(test_dataset_start,model.to('cuda'),tokenizer)

# %%
train_sentences = dataset
tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]
n = 1
train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
uni_model = Laplace(n)
uni_model.fit(train_data, padded_vocab)

# %%
n = 2
train_data = [nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
words = [word for sent in tokenized_text for word in sent]
words.extend(["<s>", "</s>"])
padded_vocab = Vocabulary(words)
bi_model = Laplace(n)
bi_model.fit(train_data, padded_vocab)

# %%
bleu = []
unigram_perplexity = []
bigram_perplexity = []
# %%
for i in range(10):
    print("Evaluating...")
    generated = text_generation(test_dataset_start,model.to('cuda'),tokenizer)

    generated_cleared = []
    for j in range(len(generated)):
        if generated[j][0] == '':
            continue
        generated_cleared.append(generated[j][0].replace(test_dataset_start[j], '').replace('<|endoftext|>','').lstrip())

    scores=[]

    for j in range(len(test_dataset)):
        reference = test_dataset_true_end[j]
        candidate = generated_cleared[j]
        # print(reference,'|',candidate,sentence_bleu(reference, candidate))
        scores.append(sentence_bleu(reference, candidate))
    bleu.append(np.mean(scores))

    test_sentences = generated_cleared
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]
    test_data, _ = padded_everygram_pipeline(n, tokenized_text)
    uni_per = []
    for test in test_data:
        per = uni_model.perplexity(test)
        if np.isfinite(per):
            uni_per.append(per)
    unigram_perplexity.append(np.mean(uni_per))

    test_data = [nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    bi_per = []
    for test in test_data:
        per = bi_model.perplexity(test)
        if np.isfinite(per):
            bi_per.append(per)
    bigram_perplexity.append(np.mean(bi_per))
    print("Training Epoch " + str(i))
    model = train(train_dataset, model, tokenizer, batch_size=1, epochs=1)
    #torch.save(model.state_dict(), "./reddit_chat_text_gen.pt",)
# %%
torch.save(model.state_dict(), "./reddit_chat_text_gen_epoch20.pt")

# %%
import matplotlib.pyplot as plt
x = np.arange(20)
plt.figure(dpi=500)
plt.plot(x, unigram_perplexity)
plt.legend(['unigram_perplexity'], loc='upper left')
# %%
plt.figure(dpi=500)
plt.plot(x, bigram_perplexity)
plt.legend(['bigram_perplexity'], loc='upper left')
# %%
plt.figure(dpi=500)
plt.plot(x, bleu)
plt.legend(['bleu'], loc='upper left')
# %%
bleu = [9.401278889349474e-232,
 1.1435979724931632e-231,
 1.1826036192617725e-231,
 1.2615024677593246e-231,
 1.243339726439941e-231,
 1.2598208850623947e-231,
 1.2545008178449618e-231,
 1.297948699489337e-231,
 1.2646619194332067e-231,
 1.2926734652633535e-231,
 1.2553102967974781e-231,
 1.2897050619978724e-231,
 1.3108489126950903e-231,
 1.313497363252657e-231,
 1.278287427876837e-231,
 1.2902687316918354e-231,
 1.2164594193258479e-231,
 1.301650396529864e-231,
 1.3516602323949798e-231,
 1.311053542034549e-231]
unigram_perplexity = [15696.683556718757,
 21256.344108011235,
 32866.07874701968,
 33594.72269658165,
 34092.23365263898,
 34620.041536859826,
 41371.80888446134,
 45456.59802191624,
 34758.072033642595,
 35334.55211442541,
 34291.65730628751,
 34660.87283512227,
 34943.11998059776,
 34161.9611904555,
 44686.883321805755,
 45123.990657272116,
 79208.67498002978,
 37160.85641490205,
 39459.1882135186,
 35216.285072723615]
bigram_perplexity = [3391.440024573186,
 3300.095107923707,
 5597.897927168722,
 5714.537618483325,
 5497.055729599305,
 5273.324351545414,
 7546.547247490587,
 7869.127446005055,
 5082.424871441698,
 5116.953085246042,
 5329.433427935666,
 5195.477506168948,
 5157.104129766492,
 5138.9173537694,
 7463.856437206803,
 7575.379373824055,
 16770.10717465791,
 5217.845229926298,
 5673.284324621978,
 5101.82148130893]
# %%
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.load_state_dict(torch.load('./reddit_chat_text_gen_epoch20.pt'))
# %%
generated = text_generation(test_dataset_start,model.to('cuda'),tokenizer)
generated_cleared = []
for j in range(len(generated)):
    if generated[j][0] == '':
        continue
    generated_cleared.append(generated[j][0].replace(test_dataset_start[j], '').replace('<|endoftext|>','').lstrip())


# %%
generated_cleared[:10]
# %%
test_dataset_start[:10]
# %%
test_dataset_true_end[:10]
# %%
