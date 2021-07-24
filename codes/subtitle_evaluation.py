import pickle
import os
import pandas as pd
import numpy as np
import ml_evaluation

os.chdir("C:/Users/VADDADISAIRAHUL/Downloads/")

successful_list = ['All Time Blockbuster','Blockbuster','Hit','Super Hit','Semi Hit','Above Average','Average']
unsuccessful_list = ['Flop','Below Average','Disaster']

file = open('indian_subtitles_final/subtitle_train.pkl','rb')
data = pickle.load(file)
file.close()

movie_names = data[0]
final_movies_names = movie_names

######################################
path_list = ['indian_movies_data_final/hindi_movies_final.csv',
             'indian_movies_data_final/Tamil_movies.csv',
             'indian_movies_data_final/telugu_movies_final.csv',
             'indian_movies_data_final/malayalam_movies_final.csv',
             'indian_movies_data_final/kannada_movies_final.csv',
             'indian_movies_data_final/other_movies_final.xlsx']
        
             
movies_list = []
verdict_list = []
roi_list = []

for path in path_list:
    data = None
    if path.endswith('.csv'):
        data = pd.read_csv(path)
    else:
        data = pd.read_excel(path)

    names = data['name']
    verdicts = data['verdict']
    budgets = data['budget']
    box_offices = data['box_office_collection_worldwide']

    for name,verdict,budget,box_office in zip(names,verdicts,budgets,box_offices):
        if (budget!=budget or budget in ["",'--','NA','N/A']) or (box_office!=box_office or box_office in ["",'--','NA','N/A']):
            continue
        else:
            movies_list.append(name)
            roi_list.append(int(box_office)/int(budget))
            verdict_list.append(verdict)
            
#########################################
subt_embedding_vectors_array = np.load("indian_subtitles_final/subtitles_embedding_vector_100d_100.npy")
for index in range(subt_embedding_vectors_array.shape[1]):
    column = subt_embedding_vectors_array[:,index]
    xrange = max(column) - min(column)
    subt_embedding_vectors_array[:,index] = (column - min(column))/xrange

X_list = []
y1_list = []
y2_list = []
included = []

for index in range(len(final_movies_names)):
    if final_movies_names[index] in movies_list:
        X_list.append(subt_embedding_vectors_array[index])
        included.append(final_movies_names[index])
        y1_list.append(roi_list[movies_list.index(final_movies_names[index])])
        verdict = verdict_list[movies_list.index(final_movies_names[index])]
        if verdict in successful_list:
            y2_list.append(1)
        else:
            y2_list.append(0)

##########################################
X = np.array(X_list)
Y1 = np.array(y1_list)
Y2 = np.array(y2_list)

# subtitle picklefile already generated

file = open('indian_subtitles_final/subtitle_data_100d_100.pkl','wb')
obj = [X,Y1,Y2,included]
data = pickle.dump(obj,file)
file.close()

##########################################
#ml_evaluation.model_evaluation(X,Y1,Y2)
