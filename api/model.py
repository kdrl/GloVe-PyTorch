import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.autograd import Variable

class GloVeClass(nn.Module):
    """
        This class provide GloVe model with some beneficial methods to utilize it.
    """
    def __init__(self, TOKENIZED_CORPUS, UNIQUE_WORD_LIST, EMBED_SIZE, CONTEXT_SIZE, X_MAX, ALPHA):
        """
            This method initialize GloVeClass with given parameters.
        
            Args:
                TOKENIZED_CORPUS(list) : list of all words in a corpus
                UNIQUE_WORD_LIST(ndarray) : list of all unique word
                EMBED_SIZE : the size of vector 
                CONTEXT_SIZE : context window size
                X_MAX : maximun x size
                ALPHA : ALPHA
        """
        super(GloVeClass, self).__init__()

        self.TOKENIZED_CORPUS = TOKENIZED_CORPUS
        self.UNIQUE_WORD_LIST = UNIQUE_WORD_LIST
        self.CONTEXT_SIZE = CONTEXT_SIZE
        self.EMBED_SIZE = EMBED_SIZE
        self.X_MAX = X_MAX
        self.ALPHA = ALPHA
        self.word_to_index = {word: index for index, word in enumerate(self.UNIQUE_WORD_LIST)}
        self.index_to_word = {index: word for index, word in enumerate(self.UNIQUE_WORD_LIST)}
        self.TOKENIZED_CORPUS_SIZE = len(self.TOKENIZED_CORPUS)
        self.UNIQUE_WORD_SIZE = len(self.UNIQUE_WORD_LIST)
        self.co_occurence_matrix = np.zeros((self.UNIQUE_WORD_SIZE, self.UNIQUE_WORD_SIZE))
        
        print("TOKENIZED_CORPUS_SIZE : ", self.TOKENIZED_CORPUS_SIZE)
        print("UNIQUE_WORD_SIZE : ", self.UNIQUE_WORD_SIZE)
        
        for i in range(self.TOKENIZED_CORPUS_SIZE):
            index = self.word_to_index[self.TOKENIZED_CORPUS[i]]
            for j in range(1, self.CONTEXT_SIZE + 1):
                if i - j > 0:
                    left_index = self.word_to_index[self.TOKENIZED_CORPUS[i-j]]
                    self.co_occurence_matrix[index, left_index] += (1.0 / j)
                    self.co_occurence_matrix[left_index, index] += (1.0 / j)
                if i + j < self.TOKENIZED_CORPUS_SIZE:
                    right_index = self.word_to_index[self.TOKENIZED_CORPUS[i + j]]
                    self.co_occurence_matrix[index, right_index] += (1.0 / j)
                    self.co_occurence_matrix[right_index, index] += (1.0 / j)
        self.co_occurence_matrix = self.co_occurence_matrix + 1.0
        
        self.in_embed = nn.Embedding(self.UNIQUE_WORD_SIZE, self.EMBED_SIZE)
        self.in_embed.weight = xavier_normal(self.in_embed.weight)
        self.in_bias = nn.Embedding(self.UNIQUE_WORD_SIZE, 1)
        self.in_bias.weight = xavier_normal(self.in_bias.weight)
        self.out_embed = nn.Embedding(self.UNIQUE_WORD_SIZE, self.EMBED_SIZE)
        self.out_embed.weight = xavier_normal(self.out_embed.weight)
        self.out_bias = nn.Embedding(self.UNIQUE_WORD_SIZE, 1)
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
        return 1 if x > self.X_MAX else (x / self.X_MAX) ** self.ALPHA

    def refill_next_batch_container(self):
        self.next_batch_container = np.append(self.next_batch_container, np.random.permutation(self.UNIQUE_WORD_SIZE*self.UNIQUE_WORD_SIZE))
        
    def next_batch(self, batch_size):
        if self.next_batch_container.size < batch_size:
            self.refill_next_batch_container()
        word_u = (self.next_batch_container[:batch_size]/self.UNIQUE_WORD_SIZE).astype(int)
        word_v = (self.next_batch_container[:batch_size]%self.UNIQUE_WORD_SIZE).astype(int)
        words_co_occurences = np.array([self.co_occurence_matrix[word_u[i], word_v[i]] for i in range(batch_size)])
        words_weights = np.array([self.weight_func(var) for var in words_co_occurences])
        self.next_batch_container = self.next_batch_container[batch_size:]
        return Variable(torch.from_numpy(word_u).cuda()), Variable(torch.from_numpy(word_v).cuda()), Variable(torch.from_numpy(words_co_occurences).cuda()).float(), Variable(torch.from_numpy(words_weights).cuda()).float()

    def embedding(self):
        self.word_embeddings_array = self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
        return self.word_embeddings_array