import numpy as np
import scipy.io
from scipy.sparse import save_npz, dok_matrix
import multiprocessing as mp
import datetime
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.autograd import Variable

def convert_pairs_to_index(word_u_index, word_v_index, UNIQUE_WORD_SIZE):
    u = min(word_u_index, word_v_index)
    v = max(word_u_index, word_v_index)
    return int((UNIQUE_WORD_SIZE + (UNIQUE_WORD_SIZE - (u - 1))) * u / 2 + (v - u))

class GloVeClass(nn.Module):
    """
        This class provide GloVe model with some beneficial methods to utilize it.
    """
    def __init__(self, TOKENIZED_CORPUS, UNIQUE_WORD_LIST, EMBED_SIZE, CONTEXT_SIZE, X_MAX, ALPHA, TOTAL_PROCESS_NUM):
        """
            This method initialize GloVeClass with given parameters.
        
            Args:
                TOKENIZED_CORPUS(list) : list of all words in a corpus
                UNIQUE_WORD_LIST(ndarray) : list of all unique word
                EMBED_SIZE : the size of vector 
                CONTEXT_SIZE : context window size
                X_MAX : maximun x size
                ALPHA : ALPHA
                TOTAL_PROCESS_NUM : TOTAL_PROCESS_NUM
        """
        super(GloVeClass, self).__init__()

        print("[Initialization Start] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
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
        
        print("TOKENIZED_CORPUS_SIZE : ", self.TOKENIZED_CORPUS_SIZE)
        print("UNIQUE_WORD_SIZE : ", self.UNIQUE_WORD_SIZE)
        
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
        
        self.total_process_num = TOTAL_PROCESS_NUM
        if TOTAL_PROCESS_NUM:
            print("TOTAL_PROCESS_NUM : ", TOTAL_PROCESS_NUM)
            queue = mp.Queue()
            ps = list()
            for i in range(self.total_process_num):
                ps.append(mp.Process(target=self.build_sub_co_occurence_matrix, args=(queue, i)))
            for p in ps:
                p.start()
            # キューから結果を回収
            for i in range(self.total_process_num):
                if i:
                    self.co_occurence_matrix += queue.get()   # キューに値が無い場合は、値が入るまで待機になる
                else:
                    self.co_occurence_matrix = queue.get()
            tries = 10
            while tries:
                try:
                    print("SAVE co_occurence_matrix @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
                    scipy.io.mmwrite('model/co_occurence_matrix.mtx', self.co_occurence_matrix)
                except IOError as e:
                    error = e
                    tries -= 1
                else:
                    break
            if not tries:
                raise error
            # np.savez('model/co_occurence_matrix.npz', self.co_occurence_matrix)
        else:
            self.co_occurence_matrix = scipy.io.mmread('model/co_occurence_matrix.mtx')
        print("[Initialization Done] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
        
    def build_sub_co_occurence_matrix(self, queue, process_num):
        print("build_sub_co_occurence_matrix in process {}(/{}) start @ {:%Y-%m-%d %H:%M:%S}".format(process_num, self.total_process_num, datetime.datetime.now()))
        # require n + ... + 1 = n(n+1)/2 space for all pairs. 
        co_occurence_matrix = dok_matrix((int((self.UNIQUE_WORD_SIZE * (self.UNIQUE_WORD_SIZE + 1)) / 2), 1), dtype=np.float32)
        # iの範囲を設定
        ini = int(self.TOKENIZED_CORPUS_SIZE * process_num / self.total_process_num)
        fin = int(self.TOKENIZED_CORPUS_SIZE * (process_num + 1) / self.total_process_num)
        for i in range(ini, fin):
            index = self.word_to_index[self.TOKENIZED_CORPUS[i]]
            for j in range(1, self.CONTEXT_SIZE + 1):
                if i - j > 0:
                    left_index = self.word_to_index[self.TOKENIZED_CORPUS[i - j]]
                    co_occurence_matrix[convert_pairs_to_index(left_index, index, self.UNIQUE_WORD_SIZE)] += (1.0 / j)
                if i + j < self.TOKENIZED_CORPUS_SIZE:
                    right_index = self.word_to_index[self.TOKENIZED_CORPUS[i + j]]
                    co_occurence_matrix[convert_pairs_to_index(right_index, index, self.UNIQUE_WORD_SIZE)] += (1.0 / j)
        queue.put(co_occurence_matrix)
        print("build_sub_co_occurence_matrix in process {}(/{}) end @ {:%Y-%m-%d %H:%M:%S}".format(process_num, self.total_process_num, datetime.datetime.now()))

    def forward(self, word_u, word_v):
        word_u_embed = self.in_embed(word_u)
        word_u_bias = self.in_bias(word_u)
        word_v_embed = self.out_embed(word_v)
        word_v_bias = self.out_bias(word_v)
        return ((word_u_embed * word_v_embed).sum(1) + word_u_bias + word_v_bias).squeeze(1)
    
    def weight_func(self, x):
        return 1 if x > self.X_MAX else (x / self.X_MAX) ** self.ALPHA

    def refill_next_batch_container(self):
        self.next_batch_container = np.append(self.next_batch_container, np.random.permutation(self.UNIQUE_WORD_SIZE * self.UNIQUE_WORD_SIZE))
        
    def next_batch(self, batch_size):
        # https://discuss.pytorch.org/t/operation-between-tensor-and-variable/1286/4
        if self.next_batch_container.size < batch_size:
            self.refill_next_batch_container()
        word_u = (self.next_batch_container[:batch_size] / self.UNIQUE_WORD_SIZE).astype(int)
        word_v = (self.next_batch_container[:batch_size] % self.UNIQUE_WORD_SIZE).astype(int)
        # + 1.e-6 -> to prevent having log(0)
        words_co_occurences = np.array(
            [self.co_occurence_matrix.get((convert_pairs_to_index(word_u[i], word_v[i], self.UNIQUE_WORD_SIZE), 0)) + 1.e-6 for i in range(batch_size)]
        )
        words_weights = np.array([self.weight_func(var) for var in words_co_occurences])
        self.next_batch_container = self.next_batch_container[batch_size:]
        return Variable(torch.from_numpy(word_u).cuda()), Variable(torch.from_numpy(word_v).cuda()), Variable(torch.from_numpy(words_co_occurences).cuda()).float(), Variable(torch.from_numpy(words_weights).cuda()).float()

    def embedding(self):
        self.word_embeddings_array = self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
        return self.word_embeddings_array
    