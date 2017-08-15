import os
import logging
import logging.handlers
import datetime
import numpy as np
import torch
from api.model import GloVeClass
from api.process import load_txt_and_tokenize

LOGFILE_PATH = "./model/20170814.log"
logger = logging.getLogger('mylogger')
fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
fileHandler = logging.FileHandler(LOGFILE_PATH)
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(fomatter)
streamHandler.setFormatter(fomatter)
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)

logger.info("[GPU setting]")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
logger.info("Done")

logger.info("[CUDA check]")
CUDA_AVAILABLE = torch.cuda.is_available()
logger.info("CUDA availability : {}".format(CUDA_AVAILABLE))
logger.info("Done")

torch.manual_seed(1)
if CUDA_AVAILABLE:
    torch.cuda.manual_seed(1)

logger.info("[Parameter setting]")
CONTEXT_SIZE = 8
EMBED_SIZE = 300
X_MAX = 100
ALPHA = 0.75
BATCH_SIZE = 4096
L_RATE = 0.05
NUM_EPOCHS = 10000
TOTAL_PROCESS_NUM = 16
LOWER_BOUND_OCCURENCE = 50
logger.info(
    "CONTEXT_SIZE = {}\nEMBED_SIZE = {}\nX_MAX = {}\nALPHA = {}\nBATCH_SIZE = {}\nL_RATE = {}\nNUM_EPOCHS = {}\nTOTAL_PROCESS_NUM = {}\nLOWER_BOUND_OCCURENCE = {}"
    .format(CONTEXT_SIZE, EMBED_SIZE, X_MAX, ALPHA, BATCH_SIZE, L_RATE, NUM_EPOCHS, TOTAL_PROCESS_NUM, LOWER_BOUND_OCCURENCE)
)
logger.info("Done")

logger.info("[Load training data]")
CORPUS_PATH = ["/data/pearl_hdd1/fukui/dataset/corpus/text8/text8"]
tokenized_corpus = load_txt_and_tokenize(CORPUS_PATH, LOWER_BOUND_OCCURENCE)
unique_word_list = np.unique(tokenized_corpus)
unique_word_list_size = unique_word_list.size
logger.info("TOKENIZED_CORPUS_SIZE : {}".format(len(tokenized_corpus)))
logger.info("UNIQUE_WORD_SIZE : {}".format(unique_word_list.size))
logger.info("Done")

logger.info("[Load model]")
GloVe = GloVeClass(tokenized_corpus, unique_word_list, EMBED_SIZE, CONTEXT_SIZE, X_MAX, ALPHA, TOTAL_PROCESS_NUM)
if CUDA_AVAILABLE:
    logger.info("with GPU")
    # GloVe = GloVe.cuda() # one gpu
    GloVe = torch.nn.DataParallel(GloVe, device_ids=[0, 1, 2, 3]).cuda()
logger.info("Done")

logger.info("[Model parameters]")
for p in GloVe.parameters():
    logger.info(p.size())
logger.info("Done")

logger.info("[Set optimizer]")
# optimizer = optim.Adam(GloVe.parameters(), L_RATE)
optimizer = torch.optim.Adagrad(GloVe.parameters(), L_RATE)
logger.info("Done")

logger.info("[Start training]")
list_for_estimation = list()
start_loop_for_estimation = 10
end_loop_for_estimation = 20

for epoch in range(NUM_EPOCHS):
    logger.info("EPOCH {} start".format(epoch + 1))
    losses = []
    update_time = ((unique_word_list_size * unique_word_list_size) // BATCH_SIZE)
    point = int(update_time / 5)
    for i in range(1, update_time + 1):

        # for epoch calculation time estimation
        if (epoch == 0) and (i >= start_loop_for_estimation):
            if (i == start_loop_for_estimation):
                time = datetime.datetime.now()
            elif (i < end_loop_for_estimation):
                list_for_estimation.append(datetime.datetime.now() - time)
                time = datetime.datetime.now()
            elif (i == end_loop_for_estimation):
                logger.info("Estimated calculation time per each epoch is {}".format(str((np.mean(list_for_estimation)) * update_time)))

        if i % point == 0:
            logger.info("{} % done".format(int(i / update_time * 100)))
        word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.module.next_batch(BATCH_SIZE)
        # word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.next_batch(self.batch_size) # when it is not in parallel
        forward_output = GloVe(word_u_variable, word_v_variable)
        loss = (torch.pow((forward_output - torch.log(words_co_occurences)), 2) * words_weights).sum()
        losses.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info("Train Epoch: {} \t Loss: {:.6f}".format(epoch + 1, np.mean(losses)))
    np.savez(
        'model/glove.npz', 
        word_embeddings_array=GloVe.module.embedding(), 
        word_to_index=GloVe.module.word_to_index,
        index_to_word=GloVe.module.index_to_word
        )
logger.info("Done")
