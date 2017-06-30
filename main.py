# import library
import os
import datetime
import numpy as np
from six.moves.urllib import request
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import torch.optim as optim
from api.model import GloVe
from api.process import load_txt_and_tokenize

# GPU setting
print("GPU setting...")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
print("Done")

# cuda check
print("Cuda check...")
cuda_available = torch.cuda.is_available()
print("cuda_available : ",cuda_available)
print("Done")

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
batch_size = 5000
l_rate = 0.05
num_epochs = 30
print("Done")

# prepare data
print("Prepare data...")
dir_path = os.path.dirname(os.path.realpath('__file__'))
corpus_path = [dir_path + '/ptb.train.txt', dir_path + '/ptb.test.txt', dir_path + '/ptb.valid.txt']
tokenized_corpus = load_txt_and_tokenize(corpus_path)
unique_word_list = np.unique(tokenized_corpus)
unique_word_list_size = unique_word_list.size
print("Done")

# Define model
print("Define model...")
glove = GloVe(tokenized_corpus, unique_word_list, embed_size, context_size, x_max, alpha)
if cuda_available:
    print("With GPU...")
    glove = torch.nn.DataParallel(glove, device_ids=[0,1,2,3]).cuda()
print("Done")

print("Model parameters...")
for p in glove.parameters():
    print(p.size())
print("Done")

print("Set optimizer...")
# optimizer = optim.Adam(glove.parameters(), l_rate)
optimizer = optim.Adagrad(glove.parameters(), l_rate)
print("Done")

print("Start training...")
print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
for epoch in range(num_epochs):
    losses = []
    for i in range((unique_word_list_size*unique_word_list_size) // batch_size):
        word_u_variable, word_v_variable, words_co_occurences, words_weights = glove.module.next_batch(batch_size)
        # word_u_variable, word_v_variable, words_co_occurences, words_weights = glove.next_batch(self.batch_size) # when it is not in parallel
        forward_output = glove(word_u_variable, word_v_variable)
        loss = (torch.pow((forward_output - torch.log(words_co_occurences)), 2) * words_weights).sum()
        losses.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i%100 == 0:
            # print("epoch{}'s proccess {} percent done.".format(epoch + 1,int(100.*i/((unique_word_list_size*unique_word_list_size) // batch_size)))) 
    print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch + 1, np.mean(losses)))
    print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    np.savez('glove.model', 
             word_embeddings_array=glove.module.embedding(), 
             word_to_index=glove.module.word_to_index,
             index_to_word=glove.module.index_to_word)
print("Done")
