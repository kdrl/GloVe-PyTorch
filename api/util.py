from sklearn.metrics.pairwise import cosine_similarity
import logging
import logging.handlers

def most_similar(word_embeddings_array, word_to_index, index_to_word, word, result_num = 1):
    data = []
    target_index = word_to_index[word]
    for i in range(word_embeddings_array.shape[0]):
        if i != target_index:
            data.append((index_to_word[i],cosine_similarity([word_embeddings_array[target_index]],[word_embeddings_array[i]])[0][0]))
    data.sort(key=lambda tup: tup[1], reverse=True)
    return data[:result_num]
