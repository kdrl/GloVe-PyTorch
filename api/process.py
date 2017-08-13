import re
import string
from collections import defaultdict
from nltk.corpus import stopwords

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    """
    # Tips for handling string in python : http://agiantmind.tistory.com/31
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

def load_txt_and_tokenize(corpus_path, lower_bound_occurence):
    """
        load corpus and remove stopwords and return tokenized corpus
        
        Args:
            corpus_path(list) : the list of path of corpus
            
        Return:
            tokenized corpus with list type
            
        Memo:
            list.remove(element) is too slow.
            Hence, in this implementation, it loads corpus file twice.
            In the first loop, it collect stopwords.
            And in the second loop, it build the list of tokenized corpus.
    """
    stop = set(stopwords.words('english'))
    tokenized_corpus = list()
    appearances = defaultdict(int)
    if type(corpus_path) is not list:
        corpus_path = [corpus_path]
    for path in corpus_path:
        with open(path) as f:
            for line in f:
                line = clean_str(line.lower().strip())
                for word in line:
                    if word not in stop:
                        appearances[word] += 1
    f.close()    
    for key, value in appearances.items():
        if value < lower_bound_occurence:
            stop.add(key)
    for path in corpus_path:
        with open(path) as f:
            for line in f:
                line = clean_str(line.lower().strip())
                for word in line:
                    if word not in stop:
                        tokenized_corpus.append(word)
    f.close()                    
    return tokenized_corpus