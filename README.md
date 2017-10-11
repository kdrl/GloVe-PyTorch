# GloVe-PyTorch
A simple PyTorch implementation for "Global Vectors for Word Representation".

## Design & Train model
See modeldesign.ipynb and main.py.

## Utilize model
See modelcheck.ipynb and below.

```python
import numpy as np
from api.util import most_similar
data = np.load('glove.model.npz')
word_embeddings_array = data['word_embeddings_array']
word_to_index = data['word_to_index'].item()
index_to_word = data['index_to_word'].item()
```


```python
print(most_similar(word_embeddings_array,word_to_index,index_to_word,"computer",result_num=5))
```

    [('software', 0.54145634), ('computers', 0.51864588), ('apple', 0.46997803), ('machines', 0.45792481), ('workstations', 0.43789768)]


