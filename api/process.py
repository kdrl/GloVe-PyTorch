from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def load_txt_and_tokenize(corpus_path):
    stop = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_corpus = list()
    if type(corpus_path) is not list:
        corpus_path = [corpus_path]
    for path in corpus_path:
        with open(path) as f:
            for line in f:
                line = tokenizer.tokenize(line.lower().strip())
                for word in line:
                    if word not in stop and len(word) > 1:
                        tokenized_corpus.append(word)
    return tokenized_corpus