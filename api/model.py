import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.autograd import Variable

class GloVeClass(nn.Module):
    """
        This class provide GloVe model with some beneficial methods to utilize it.
    """
    def __init__(self, tokenized_corpus, unique_word_list, embed_size, context_size, x_max, alpha):
        """
            This method initialize GloVeClass with given parameters.
        
            Args:
                tokenized_corpus(list) : list of all words in a corpus
                unique_word_list(ndarray) : list of all unique word
                embed_size : the size of vector 
                context_size : context window size
                x_max : maximun x size
                alpha : alpha
        """
        super(GloVeClass, self).__init__()

        self.TOKENIZED_CORPUS = tokenized_corpus
        self.UNIQUE_WORD_LIST = unique_word_list
        self.CONTEXT_SIZE = context_size
        self.EMBED_SIZE = embed_size
        self.X_MAX = x_max
        self.ALPHA = alpha
        
        self.word_to_index = {word: index for index, word in enumerate(self.unique_word_list)}
        self.index_to_word = {index: word for index, word in enumerate(self.unique_word_list)}
        self.tokenized_corpus_size = len(self.tokenized_corpus)
        self.unique_word_size = len(self.unique_word_list)
        
        print("tokenized_corpus_size : ",self.tokenized_corpus_size)
        print("unique_word_size : ",self.unique_word_size)
        
        self.co_occurence_matrix = np.zeros((self.unique_word_size, self.unique_word_size))
        for i in range(self.tokenized_corpus_size):
            index = self.word_to_index[self.tokenized_corpus[i]]
            for j in range(1, self.context_size + 1):
                if i-j > 0:
                    left_index = self.word_to_index[self.tokenized_corpus[i-j]]
                    self.co_occurence_matrix[index, left_index] += (1.0/j)
                    self.co_occurence_matrix[left_index, index] += (1.0/j)
                if i+j < self.tokenized_corpus_size:
                    right_index = self.word_to_index[self.tokenized_corpus[i+j]]
                    self.co_occurence_matrix[index, right_index] += (1.0/j)
                    self.co_occurence_matrix[right_index, index] += (1.0/j)
        self.co_occurence_matrix = self.co_occurence_matrix + 1.0
        
        self.in_embed = nn.Embedding(self.unique_word_size, self.embed_size)
        self.in_embed.weight = xavier_normal(self.in_embed.weight)
        self.in_bias = nn.Embedding(self.unique_word_size, 1)
        self.in_bias.weight = xavier_normal(self.in_bias.weight)
        self.out_embed = nn.Embedding(self.unique_word_size, self.embed_size)
        self.out_embed.weight = xavier_normal(self.out_embed.weight)
        self.out_bias = nn.Embedding(self.unique_word_size, 1)
        self.out_bias.weight = xavier_normal(self.out_bias.weight)
        
        self.next_batch_container = np.array([]).astype(int)
        self.word_embeddings_array = None

    def forward(self, word_u, word_v):
        word_u_embed = self.in_embed(word_u)
        word_u_bias = self.in_bias(word_u)
        word_v_embed = self.out_embed(word_v)
        word_v_bias = self.out_bias(word_v)
        return ((word_u_embed * word_v_embed).sum(1) + word_u_bias + word_v_bias).squeeze(1)
    
    def weight_func(self, x):
        return 1 if x > self.x_max else (x / self.x_max) ** self.alpha

    def refill_next_batch_container(self):
        self.next_batch_container = np.append(self.next_batch_container, np.random.permutation(self.unique_word_size*self.unique_word_size))
        
    def next_batch(self, batch_size):
        if self.next_batch_container.size < batch_size:
            self.refill_next_batch_container()
        word_u = list()
        word_v = list()
        for i in self.next_batch_container[:batch_size]:
            word_u.append(int(i/self.unique_word_size))
            word_v.append(i%self.unique_word_size)
        word_u = np.array(word_u)
        word_v = np.array(word_v)
        words_co_occurences = np.array([self.co_occurence_matrix[word_u[i], word_v[i]] for i in range(batch_size)])
        words_weights = np.array([self.weight_func(var) for var in words_co_occurences])
        words_co_occurences = Variable(torch.from_numpy(words_co_occurences).cuda()).float()
        words_weights = Variable(torch.from_numpy(words_weights).cuda()).float()
        word_u = Variable(torch.from_numpy(word_u).cuda())
        word_v = Variable(torch.from_numpy(word_v).cuda())
        self.next_batch_container = self.next_batch_container[batch_size:]
        return word_u, word_v, words_co_occurences, words_weights

    def embedding(self):
        self.word_embeddings_array = self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
        return self.word_embeddings_array