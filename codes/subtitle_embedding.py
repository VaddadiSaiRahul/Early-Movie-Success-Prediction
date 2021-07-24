import pickle
import os
from gensim.models.doc2vec import Doc2Vec
import numpy as np

os.chdir("C:/Users/VADDADISAIRAHUL/Downloads/indian_subtitles_final/")

file = open('subtitle_train.pkl','rb')
data = pickle.load(file)
file.close()

all_subt_words_list = data[1]

model = Doc2Vec.load("subt2vec100d_10.model")
subt_embedding_vectors_list = []

for subt_words_list in all_subt_words_list:
    subt_embedding_vectors_list.append(model.infer_vector(subt_words_list))

subt_embedding_vectors_array = np.array(subt_embedding_vectors_list)

np.save("subtitles_embedding_vector_100d_10",subt_embedding_vectors_array)
