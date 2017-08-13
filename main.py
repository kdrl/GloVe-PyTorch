import os
import datetime
import numpy as np
import torch
from api.model import GloVeClass
from api.process import load_txt_and_tokenize
from api.util import LoggerClass

logger = LoggerClass("./model/20170814.log")

logger.print_and_log("[GPU setting]")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
logger.print_and_log("Done")

logger.print_and_log("[CUDA check]")
CUDA_AVAILABLE = torch.cuda.is_available()
logger.print_and_log("CUDA availability : ", CUDA_AVAILABLE)
logger.print_and_log("Done")

torch.manual_seed(1)
if CUDA_AVAILABLE:
    torch.cuda.manual_seed(1)

logger.print_and_log("[Parameter setting]")
CONTEXT_SIZE = 8
EMBED_SIZE = 300
X_MAX = 100
ALPHA = 0.75
BATCH_SIZE = 4096
L_RATE = 0.05
NUM_EPOCHS = 10000
TOTAL_PROCESS_NUM = 16
LOWER_BOUND_OCCURENCE = 50
logger.print_and_log(
    "CONTEXT_SIZE = {}\nEMBED_SIZE = {}\nX_MAX = {}\nALPHA = {}\nBATCH_SIZE = {}\nL_RATE = {}\nNUM_EPOCHS = {}\nTOTAL_PROCESS_NUM = {}\nLOWER_BOUND_OCCURENCE = {}"
    .format(CONTEXT_SIZE, EMBED_SIZE, X_MAX, ALPHA, BATCH_SIZE, L_RATE, NUM_EPOCHS, TOTAL_PROCESS_NUM, LOWER_BOUND_OCCURENCE)
)
logger.print_and_log("Done")

logger.print_and_log("[Load training data]")
CORPUS_PATH = ["/data/pearl_hdd1/fukui/dataset/corpus/text8/text8"]
tokenized_corpus = load_txt_and_tokenize(CORPUS_PATH, LOWER_BOUND_OCCURENCE)
unique_word_list = np.unique(tokenized_corpus)
unique_word_list_size = unique_word_list.size
logger.print_and_log("TOKENIZED_CORPUS_SIZE : ", len(load_txt_and_tokenize))
logger.print_and_log("UNIQUE_WORD_SIZE : ", unique_word_list.size)
logger.print_and_log("Done")

logger.print_and_log("[Load model]")
GloVe = GloVeClass(tokenized_corpus, unique_word_list, EMBED_SIZE, CONTEXT_SIZE, X_MAX, ALPHA, TOTAL_PROCESS_NUM)
if CUDA_AVAILABLE:
    logger.print_and_log("with GPU")
    # GloVe = GloVe.cuda() # one gpu
    GloVe = torch.nn.DataParallel(GloVe, device_ids=[0, 1, 2, 3]).cuda()
logger.print_and_log("Done")

logger.print_and_log("[Model parameters]")
for p in GloVe.parameters():
    logger.print_and_log(p.size())
logger.print_and_log("Done")

logger.print_and_log("[Set optimizer]")
# optimizer = optim.Adam(GloVe.parameters(), L_RATE)
optimizer = torch.optim.Adagrad(GloVe.parameters(), L_RATE)
logger.print_and_log("Done")

logger.print_and_log("[Start training]")
for epoch in range(NUM_EPOCHS):
    logger.print_and_log("EPOCH {} start".format(epoch + 1))
    losses = []
    update_time = ((unique_word_list_size * unique_word_list_size) // BATCH_SIZE)
    point = int(update_time / 5)
    for i in range(1, update_time + 1):
        if ((epoch == 0) and (i == 1)):
            time1 = datetime.datetime.now()
        if ((epoch == 0) and (i == 2)):
            time2 = datetime.datetime.now()
            logger.print_and_log("Estimated calculation time per each epoch is {}".format(str((time2 - time1) * update_time)))
        if i % point == 0:
            logger.print_and_log("{} % done".format(int(i / update_time * 100))
            word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.module.next_batch(BATCH_SIZE)
            # word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.next_batch(self.batch_size) # when it is not in parallel
            forward_output = GloVe(word_u_variable, word_v_variable)
            loss = (torch.pow((forward_output - torch.log(words_co_occurences)), 2) * words_weights).sum()
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    logger.print_and_log("Train Epoch: {} \t Loss: {:.6f}".format(epoch + 1, np.mean(losses))
    np.savez('model/glove.npz', 
             word_embeddings_array=GloVe.module.embedding(), 
             word_to_index=GloVe.module.word_to_index,
             index_to_word=GloVe.module.index_to_word)
logger.print_and_log("Done")
