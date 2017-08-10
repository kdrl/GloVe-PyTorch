import re
import string
from nltk.corpus import stopwords

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().split()

def load_txt_and_tokenize(corpus_path):
    stop = set(stopwords.words('english'))
    alphabets = list(string.ascii_lowercase)
    alphabets.remove('a')
    alphabets.remove('i')
    stop.update(alphabets)
    tokenized_corpus = list()
    if type(corpus_path) is not list:
        corpus_path = [corpus_path]
    for path in corpus_path:
        with open(path) as f:
            for line in f:
                line = clean_str(line.lower().strip())
                for word in line:
                    if word not in stop:
                        tokenized_corpus.append(word)
    return tokenized_corpus