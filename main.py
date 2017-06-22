# import library
import os
import matplotlib
import numpy as np
import collections
from six.moves.urllib import request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant
from sklearn.metrics.pairwise import cosine_similarity

# GPU setting
print("GPU setting...")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

# cuda check
print("Cuda check...")
cuda_available = torch.cuda.is_available()
print("cuda_available : ",cuda_available)

# static seed
torch.manual_seed(1)
if cuda_available:
    torch.cuda.manual_seed(1)

# Set parameters
print("Set parameters...")
context_size = 10
embed_size = 500
x_max = 100
alpha = 0.75
batch_size = 50
l_rate = 0.001
num_epochs = 30

# define methods and classes
print("Define methods and classes...")
def clean_str(string):
    # Tips for handling string in python : http://agiantmind.tistory.com/31
    string = string.lower()
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def weight_func(x):
    return 1 if x > x_max else (x / x_max) ** alpha

def next_batch(batch_size,word_u,word_v):

    words_co_occurences = np.array([co_occurence_matrix[word_u[i], word_v[i]] for i in range(batch_size)])
    words_weights = np.array([weight_func(var) for var in words_co_occurences])
    
    words_co_occurences = Variable(torch.from_numpy(words_co_occurences).cuda()).float()
    words_weights = Variable(torch.from_numpy(words_weights).cuda()).float()

    word_u = Variable(torch.from_numpy(word_u).cuda())
    word_v = Variable(torch.from_numpy(word_v).cuda())

    return word_u, word_v, words_co_occurences, words_weights

def most_similar(word_embeddings_array, word, result_num = 1):
    data = []
    num = word_embeddings_array.shape[0]
    target_index = word_to_index[word]
    for i in range(num):
        if i != target_index:
            data.append((index_to_word[i],cosine_similarity([word_embeddings_array[target_index]],[word_embeddings_array[i]])[0][0]))
    data.sort(key=lambda tup: tup[1], reverse=True)
    return data[:result_num]

class GloVe(nn.Module):
    def __init__(self, num_classes, embed_size):

        super(GloVe, self).__init__()

        self.num_classes = num_classes
        self.embed_size = embed_size
        
        self.in_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.in_embed.weight = xavier_normal(self.in_embed.weight)

        self.in_bias = nn.Embedding(self.num_classes, 1)
        self.in_bias.weight = xavier_normal(self.in_bias.weight)

        self.out_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.out_embed.weight = xavier_normal(self.out_embed.weight)

        self.out_bias = nn.Embedding(self.num_classes, 1)
        self.out_bias.weight = xavier_normal(self.out_bias.weight)

    def forward(self, word_u, word_v):

        word_u_embed = self.in_embed(word_u)
        word_u_bias = self.in_bias(word_u)
        word_v_embed = self.out_embed(word_v)
        word_v_bias = self.out_bias(word_v)
        
        return ((word_u_embed * word_v_embed).sum(1) + word_u_bias + word_v_bias).squeeze(1)
    
    def embeddings(self):
        return self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
    
# prepare data
print("Prepare data...")

stop = set(stopwords.words('english'))
word_list = list()

with open('ptb.train.txt') as f:
    for line in f:
        line = clean_str(line)
        for word in line.split():
            if word not in stop and len(word) > 1:
                word_list.append(word)
                
vocab = np.unique(word_list)
word_to_index = {word: index for index, word in enumerate(vocab)}
index_to_word = {index: word for index, word in enumerate(vocab)}
word_list_size = len(word_list)
vocab_size = len(vocab)

print("word_list_size", word_list_size)
print("vocab_size", vocab_size)

# Construct co-occurence matrix
print("Construct co-occurence matrix...")
co_occurence_matrix = np.zeros((vocab_size, vocab_size))
for i in range(word_list_size):
    for j in range(1, context_size + 1):
        index = word_to_index[word_list[i]]
        if i-j > 0:
            left_index = word_to_index[word_list[i-j]]
            co_occurence_matrix[index, left_index] += (1.0/j)
            co_occurence_matrix[left_index, index] += (1.0/j)
        if i+j < word_list_size:
            right_index = word_to_index[word_list[i+j]]
            co_occurence_matrix[index, right_index] += (1.0/j)
            co_occurence_matrix[right_index, index] += (1.0/j)

co_occurence_matrix = co_occurence_matrix + 1.0
[num_classes, _] = co_occurence_matrix.shape
   
# Define model
print("Define model...")
glove = GloVe(num_classes, embed_size)
if cuda_available:
    print("With GPU...")
    glove = torch.nn.DataParallel(glove, device_ids=[0,1,2,3]).cuda()
      
print("Model parameters...")
for p in glove.parameters():
    print(p.size())
      
print("Set optimizer...")
# optimizer = optim.Adam(glove.parameters(), l_rate)
optimizer = optim.Adagrad(glove.parameters(), 0.05)
 
print("Start training...")
for epoch in range(num_epochs):
    losses = []
    random_word_u_indexes = np.random.permutation(num_classes)
    for word_u_position in range(0, num_classes, batch_size):
        word_u = random_word_u_indexes[word_u_position:(word_u_position + batch_size) if (word_u_position+batch_size) < num_classes else -1]
        cycle_size = word_u.shape[0]
        random_word_v_indexes = np.random.permutation(num_classes)
        for word_v_position in range(0, num_classes, cycle_size):
            word_v = random_word_v_indexes[word_v_position:(word_v_position + cycle_size) if (word_v_position+cycle_size) < num_classes else -1]
            if cycle_size != word_v.shape[0]:
                continue
            word_u_variable, word_v_variable, words_co_occurences, words_weights = next_batch(cycle_size, word_u, word_v)
            forward_output = glove(word_u_variable, word_v_variable)
            loss = (torch.pow((forward_output - torch.log(words_co_occurences)), 2) * words_weights).sum()
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
      
    print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch + 1, np.mean(losses)))
    torch.save(glove.module.state_dict(), "./glove.model")