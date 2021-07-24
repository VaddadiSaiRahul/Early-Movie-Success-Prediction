import os
import pickle
import numpy as np
import ml_evaluation

os.chdir("C:/Users/VADDADISAIRAHUL/Downloads/")

# open movie data pickle file
file1 = open('indian_movies_data_final/movie_data_final.pkl','rb')
data1 = pickle.load(file1)
file1.close()

# open subtitle data pickle file
file3 = open('indian_subtitles_final/subtitle_data_100d_10.pkl','rb')
data3 = pickle.load(file3)
file3.close()

subtitle_data,subtitle_data_movies = data3, data3[3]
movie_data,movie_data_movies = data1, data1[3]

X_movie = movie_data[0]# except roi and verdict
Y1_movie = movie_data[1]#** roi same for subtitle or movie data (since movies_Data>subtitles_data hence taking moviedata is enough)
Y2_movie = movie_data[2]#** verdict same for subtitle or movie data (since movies_Data>subtitles_data hence taking moviedata is enough)
X_subt = subtitle_data[0]# doc2vec matrix

movies_intersect = list(set(subtitle_data_movies).intersection(set(movie_data_movies)))
movies_intersect.sort()

X = []
Y1 = []
Y2 = []

for movie in movies_intersect:
    index1 = subtitle_data_movies.index(movie)
    index2 = movie_data_movies.index(movie)
    merged_X = np.concatenate((X_subt[index1],X_movie[index2]))
    X.append(merged_X);Y1.append(Y1_movie[index2]);Y2.append(int(Y2_movie[index2]))

data = np.array(X)
target1 = np.array(Y1)
target2 = np.array(Y2)
print("Movie Data Shape :",X_movie.shape)
print("Subtitles Data Shape :",X_subt.shape)
print("Merged Data Shape :",data.shape)
print("ROI data shape :",target1.shape)
print("Verdict data shape :",target2.shape)
ml_evaluation.model_evaluation(data,target1,target2)
