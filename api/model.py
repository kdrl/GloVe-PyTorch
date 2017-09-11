import numpy as np
import scipy.io
from scipy.sparse import save_npz, load_npz, coo_matrix
import multiprocessing as mp
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.autograd import Variable

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

        print("[Initialization Start]")
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

        self.in_embed = nn.Embedding(self.UNIQUE_WORD_SIZE, self.EMBED_SIZE)
        self.in_embed.weight = xavier_normal(self.in_embed.weight)
        self.in_bias = nn.Embedding(self.UNIQUE_WORD_SIZE, 1)
        self.in_bias.weight = xavier_normal(self.in_bias.weight)
        self.out_embed = nn.Embedding(self.UNIQUE_WORD_SIZE, self.EMBED_SIZE)
        self.out_embed.weight = xavier_normal(self.out_embed.weight)
        self.out_bias = nn.Embedding(self.UNIQUE_WORD_SIZE, 1)
        self.out_bias.weight = xavier_normal(self.out_bias.weight)
        
        self.word_embeddings_array = None
        self.word_u_candidate = np.arange(self.UNIQUE_WORD_SIZE)
        self.word_v_candidate = np.arange(self.UNIQUE_WORD_SIZE)
        
        self.total_process_num = TOTAL_PROCESS_NUM
        if TOTAL_PROCESS_NUM:
            print("Build co-occurence matrix with multiprocess")
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
                    col += queue.get()   # キューに値が無い場合は、値が入るまで待機になる
                else:
                    col = queue.get()
            for p in ps:
                p.terminate()
            col = np.array(col, dtype = np.int64)
            self.co_occurence_matrix = coo_matrix(
                (np.ones(col.size, dtype = np.int64), (np.zeros(col.size, dtype = np.int64), col)), 
                shape=(1, int((self.UNIQUE_WORD_SIZE * (self.UNIQUE_WORD_SIZE + 1)) / 2)),
                dtype = np.int64
            )
            print("Done")                             
            tries = 10
            while tries:
                try:
                    print("SAVE co_occurence_matrix")
                    # scipy.io.mmwrite('model/co_occurence_matrix.mtx', self.co_occurence_matrix)
                    save_npz('model/co_occurence_matrix.npz', self.co_occurence_matrix)
                    print("Done")
                except IOError as e:
                    print("IOError happened")
                    error = e
                    tries -= 1
                else:
                    break
            if not tries:
                print("Fail to saving matrix due to IOError")
                raise error
        else:
            print("Load co-occurence matrix")
            # self.co_occurence_matrix = scipy.io.mmread('model/co_occurence_matrix.mtx')
            self.co_occurence_matrix = load_npz('model/co_occurence_matrix.npz')
            print("Done")
        self.co_occurence_matrix = self.co_occurence_matrix.todense()
        print("[Initialization Done]")
        
    def build_sub_co_occurence_matrix(self, queue, process_num):
        col = list()
        # iの範囲を設定
        ini = int(self.TOKENIZED_CORPUS_SIZE * process_num / self.total_process_num)
        fin = int(self.TOKENIZED_CORPUS_SIZE * (process_num + 1) / self.total_process_num)
        for i in range(ini, fin):
            index = self.word_to_index[self.TOKENIZED_CORPUS[i]]
            for j in range(1, self.CONTEXT_SIZE + 1):
                if i - j > 0:
                    left_index = self.word_to_index[self.TOKENIZED_CORPUS[i - j]]
                    col.append(self.convert_pairs_to_index(left_index, index))
                if i + j < self.TOKENIZED_CORPUS_SIZE:
                    right_index = self.word_to_index[self.TOKENIZED_CORPUS[i + j]]
                    col.append(self.convert_pairs_to_index(right_index, index))
        queue.put(col)
    
    def weight_func(self, x):
        return 1 if x > self.X_MAX else (x / self.X_MAX) ** self.ALPHA

    def convert_pairs_to_index(self, word_u_index, word_v_index):
        u = min(word_u_index, word_v_index)
        v = max(word_u_index, word_v_index)
        # return int((UNIQUE_WORD_SIZE + (UNIQUE_WORD_SIZE - (u - 1))) * u / 2 + (v - u))
        return int((2 * self.UNIQUE_WORD_SIZE - u + 1) * u / 2 + v - u)

    def next_batch(self, batch_size):
        word_u = np.random.choice(self.word_u_candidate, size=batch_size)
        word_v = np.random.choice(self.word_v_candidate, size=batch_size)
        # https://discuss.pytorch.org/t/operation-between-tensor-and-variable/1286/4
        #word_u = np.random.randint(self.UNIQUE_WORD_SIZE, size=batch_size)
        #word_v = np.random.randint(self.UNIQUE_WORD_SIZE, size=batch_size)
        # + 1 -> to prevent having log(0)
        words_co_occurences = np.array(
            [self.co_occurence_matrix[0, self.convert_pairs_to_index(word_u[i], word_v[i])] + 1 for i in range(batch_size)]
        )
        words_weights = np.array([self.weight_func(var) for var in words_co_occurences])
        return Variable(torch.from_numpy(word_u).cuda()), Variable(torch.from_numpy(word_v).cuda()), Variable(torch.from_numpy(words_co_occurences).cuda()).float(), Variable(torch.from_numpy(words_weights).cuda()).float()

    def forward(self, word_u, word_v):
        word_u_embed = self.in_embed(word_u)
        word_u_bias = self.in_bias(word_u)
        word_v_embed = self.out_embed(word_v)
        word_v_bias = self.out_bias(word_v)
        return ((word_u_embed * word_v_embed).sum(1) + word_u_bias + word_v_bias).squeeze(1)
    
    def embedding(self):
        self.word_embeddings_array = self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
        return self.word_embeddings_array
    