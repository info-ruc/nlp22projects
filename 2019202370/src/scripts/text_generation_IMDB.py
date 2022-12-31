# %%
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.lm import Vocabulary
from utils import *
dataset = load_dataset("imdb")

#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

# %%
dataset = dataset['train']['text']
train_dataset = dataset[:-100]
test_dataset = dataset[-100:]

train_dataset = Entries(dataset, tokenizer, truncate=False)  

test_dataset_true_end = [' '.join(x.split()[math.floor(len(x.split())*0.8):]) for x in test_dataset]
test_dataset_start = [' '.join(x.split()[:math.floor(len(x.split())*0.8)]) for x in test_dataset]

# %%
train_sentences = dataset
tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]
n = 1
train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
uni_model = Laplace(n)
uni_model.fit(train_data, padded_vocab)

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

# %%
torch.save(model.state_dict(), "./IMDB_text_gen_epoch20.pt")

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
bleu =[1.0664451186521016e-231,
 1.0946993856008654e-231,
 1.1588421910302205e-231,
 1.140953148006835e-231,
 1.1318018707135604e-231,
 1.1181808186763263e-231,
 1.1569687532837175e-231,
 1.1409471206378317e-231,
 1.181348710548114e-231,
 1.131214939110326e-231,
 1.1771590978616554e-231,
 1.1382748107474999e-231,
 1.1875560270232964e-231,
 1.1223520421421013e-231,
 1.111868751773515e-231,
 1.18843226715473e-231,
 1.194770089480644e-231,
 1.1531531418321022e-231,
 1.1454913023319787e-231,
 1.1825638296510522e-231]
unigram_perplexity = [10773.512764932499,
 45573.773514899134,
 30788.202142452483,
 47952.009703126925,
 65902.32110062466,
 29053.880704497435,
 12942.115679182647,
 50337.66531090648,
 39003.94410429533,
 67931.0814222864,
 48670.07899710925,
 46303.233279370324,
 50410.58619406223,
 31710.43776611654,
 67183.44856780912,
 15991.728071151794,
 52785.85209173565,
 84723.44805240237,
 51302.75941049132,
 34004.93217515094]
bigram_perplexity = [1672.978823308028,
 3873.99928918528,
 2466.0176835903717,
 4001.161220558877,
 5144.927465701785,
 2602.6526117873764,
 1113.6426091678727,
 3869.954061421996,
 3881.3537508702375,
 6029.930239514411,
 3866.926263610346,
 3823.428095562332,
 4128.60219174894,
 3128.654874884477,
 5295.708097424466,
 1547.8056122807216,
 4188.015660935993,
 6712.806410238624,
 4238.81200608842,
 2690.466926560107]
# %%
# %%
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.load_state_dict(torch.load('./IMDB_text_gen_epoch20.pt'))
# %%
generated = text_generation(test_dataset_start,model.to('cuda'),tokenizer)
generated_cleared = []
for j in range(len(generated)):
    if generated[j][0] == '':
        continue
    generated_cleared.append(generated[j][0].replace(test_dataset_start[j], '').replace('<|endoftext|>','').lstrip())


# %%
generated_cleared[:5]
# %%
test_dataset_start[:5]
# %%
test_dataset_true_end[:5]
# %%
