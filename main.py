import os
import datetime
import numpy as np
import torch
from api.model import GloVeClass
from api.process import load_txt_and_tokenize

print("[GPU setting] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
print("Done @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))

print("[CUDA check] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
CUDA_AVAILABLE = torch.cuda.is_available()
print("CUDA availability : ", CUDA_AVAILABLE)
print("Done @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))

torch.manual_seed(1)
if CUDA_AVAILABLE:
    torch.cuda.manual_seed(1)

print("[Parameter setting] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
CONTEXT_SIZE = 5
EMBED_SIZE = 300
X_MAX = 100
ALPHA = 0.75
BATCH_SIZE = 8192
L_RATE = 0.05
NUM_EPOCHS = 10000
TOTAL_PROCESS_NUM = 16
print(
    "CONTEXT_SIZE = {}\nEMBED_SIZE = {}\nX_MAX = {}\nALPHA = {}\nBATCH_SIZE = {}\nL_RATE = {}\nNUM_EPOCHS = {}\nTOTAL_PROCESS_NUM = {}"
    .format(CONTEXT_SIZE, EMBED_SIZE, X_MAX, ALPHA, BATCH_SIZE, L_RATE, NUM_EPOCHS, TOTAL_PROCESS_NUM)
)
print("Done @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))

print("[Load training data] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
CORPUS_PATH = ["/data/pearl_hdd1/fukui/dataset/corpus/text8/text8"]
tokenized_corpus = load_txt_and_tokenize(CORPUS_PATH)
unique_word_list = np.unique(tokenized_corpus)
unique_word_list_size = unique_word_list.size
print("Done @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))

print("[Load model] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
GloVe = GloVeClass(tokenized_corpus, unique_word_list, EMBED_SIZE, CONTEXT_SIZE, X_MAX, ALPHA, TOTAL_PROCESS_NUM)
if CUDA_AVAILABLE:
    print("with GPU")
    # GloVe = GloVe.cuda() # one gpu
    GloVe = torch.nn.DataParallel(GloVe, device_ids=[0, 1, 2, 3]).cuda()
print("Done @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))

print("[Model parameters] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
for p in GloVe.parameters():
    print(p.size())
print("Done @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))

print("[Set optimizer] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
# optimizer = optim.Adam(GloVe.parameters(), L_RATE)
optimizer = torch.optim.Adagrad(GloVe.parameters(), L_RATE)
print("Done @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))

print("[Start training] @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
for epoch in range(NUM_EPOCHS):
    print("EPOCH {} start".format(epoch + 1))
    losses = []
    update_time = ((unique_word_list_size * unique_word_list_size) // BATCH_SIZE)
    point = int(update_time / 5)
    for i in range(1, update_time + 1):
        
        if ((epoch == 0) and (i == 1)):
            time1 = datetime.datetime.now()
        elif ((epoch == 0) and (i == 2)):
            time2 = datetime.datetime.now()
            print("Estimated calculation time per each epoch is {}".format(str((time2 - time1) * update_time)))
            
        if i % point == 0:
            print("{} % done @ {:%Y-%m-%d %H:%M:%S}".format(int(i / update_time * 100), datetime.datetime.now()))
            
        word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.module.next_batch(BATCH_SIZE)
        # word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.next_batch(self.batch_size) # when it is not in parallel
        forward_output = GloVe(word_u_variable, word_v_variable)
        loss = (torch.pow((forward_output - torch.log(words_co_occurences)), 2) * words_weights).sum()
        losses.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Train Epoch: {} \t Loss: {:.6f} @ {:%Y-%m-%d %H:%M:%S}".format(epoch + 1, np.mean(losses), datetime.datetime.now()))
    np.savez('model/glove.npz', 
             word_embeddings_array=GloVe.module.embedding(), 
             word_to_index=GloVe.module.word_to_index,
             index_to_word=GloVe.module.index_to_word)
print("Done @ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
