import os
import datetime
import numpy as np
import torch
from api.model import GloVeClass
from api.process import load_txt_and_tokenize

print("[GPU setting]")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
print("Done")

print("[CUDA check]")
CUDA_AVAILABLE = torch.cuda.is_available()
print("CUDA availability : ", CUDA_AVAILABLE)
print("Done")

torch.manual_seed(1)
if CUDA_AVAILABLE:
    torch.cuda.manual_seed(1)

print("[Parameter setting]")
CONTEXT_SIZE = 10
EMBED_SIZE = 500
X_MAX = 100
ALPHA = 0.75
BATCH_SIZE = 5000
L_RATE = 0.05
NUM_EPOCHS = 30
print(
    "CONTEXT_SIZE = {}\nEMBED_SIZE = {}\nX_MAX = {}\nALPHA = {}\nBATCH_SIZE = {}\nL_RATE = {}\nNUM_EPOCHS = {}"
    .format(CONTEXT_SIZE, EMBED_SIZE, X_MAX, ALPHA, BATCH_SIZE, L_RATE, NUM_EPOCHS)
)
print("Done")

print("[Load training data]")
CORPUS_PATH = ["/data/pearl_hdd1/fukui/dataset/corpus/text8/text8"]
tokenized_corpus = load_txt_and_tokenize(CORPUS_PATH)
unique_word_list = np.unique(tokenized_corpus)
unique_word_list_size = unique_word_list.size
print("Done")

print("[Load model]")
GloVe = GloVeClass(tokenized_corpus, unique_word_list, EMBED_SIZE, CONTEXT_SIZE, X_MAX, ALPHA)
if CUDA_AVAILABLE:
    print("with GPU")
    GloVe = torch.nn.DataParallel(GloVe, device_ids=[0, 1, 2, 3]).cuda()
print("Done")

print("[Model parameters]")
for p in GloVe.parameters():
    print(p.size())
print("Done")

print("[Set optimizer]")
# optimizer = optim.Adam(GloVe.parameters(), L_RATE)
optimizer = torch.optim.Adagrad(GloVe.parameters(), L_RATE)
print("Done")

print("[Start training]")
print("@ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
for epoch in range(NUM_EPOCHS):
    losses = []
    for i in range((unique_word_list_size*unique_word_list_size) // BATCH_SIZE):
        word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.module.next_batch(BATCH_SIZE)
        # word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.next_batch(self.batch_size) # when it is not in parallel
        forward_output = GloVe(word_u_variable, word_v_variable)
        loss = (torch.pow((forward_output - torch.log(words_co_occurences)), 2) * words_weights).sum()
        losses.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Train Epoch: {} \t Loss: {:.6f}".format(epoch + 1, np.mean(losses)))
    print("@ {:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
    np.savez('model/glove.npz', 
             word_embeddings_array=GloVe.module.embedding(), 
             word_to_index=GloVe.module.word_to_index,
             index_to_word=GloVe.module.index_to_word)
print("Done")
