import pickle
import os
import numpy as np
import ml_evaluation

os.chdir("C:/Users/VADDADISAIRAHUL/Downloads/indian_movies_data_final/")

file = open('movie_data_1.pkl','rb')
data = pickle.load(file)
file.close()

attributes = data[0]
dataarray = data[2]
numerical_attributes = ['runtime','budget','box_office_collection_worldwide',
                        'actors','music','director','writer','producer','roi']


# min-max scaling
for attribute in numerical_attributes:
    index = attributes.index(attribute)
    xrange = max(dataarray[:,index]) - min(dataarray[:,index])
    dataarray[:,index] = (dataarray[:,index] - min(dataarray[:,index]))/xrange

dataarray_cleaned = np.delete(dataarray,attributes.index('box_office_collection_worldwide'),1)

X = dataarray_cleaned[:,:-2]
Y1 = dataarray_cleaned[:,-2]
Y2 = dataarray_cleaned[:,-1]

# movie pickle file already generated
file = open('movie_data_final.pkl','wb')
obj = [X,Y1,Y2,data[1]]
pickle.dump(obj,file)
file.close()
#ml_evaluation.model_evaluation(X,Y1,Y2)

