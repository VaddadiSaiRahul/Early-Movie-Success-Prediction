import os
from gensim.models.doc2vec import Doc2Vec
import pickle

os.chdir("C:/Users/VADDADISAIRAHUL/Downloads/indian_subtitles_final/")

def train_subt2vec(docs, vec_size = 100, alpha = 0.025):

    model = Doc2Vec(vector_size=vec_size,alpha=alpha,min_alpha=0.0001,min_count=5,dm=0)
    model.build_vocab(docs)
    
    model.train(docs,total_examples=model.corpus_count,epochs=10)
    model.min_alpha = model.alpha

    return model

def main():
    file = open('subtitle_train.pkl','rb')
    data = pickle.load(file)
    file.close()

    docs = data[2]
    
    model = train_subt2vec(docs)
    model.save("subt2vec100d_10.model")
    print("Model Saved")
    

if __name__=='__main__':
    main()
