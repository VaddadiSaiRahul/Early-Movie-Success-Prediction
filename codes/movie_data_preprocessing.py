import os
import pandas as pd
import numpy as np
import math
import pickle

os.chdir('C:/Users/VADDADISAIRAHUL/Downloads/indian_movies_data_final/')

successful_list = ['All Time Blockbuster','Blockbuster','Hit','Super Hit','Semi Hit','Above Average','Average']
unsuccessful_list = ['Flop','Below Average','Disaster']

def is_null_record(data_dict):
    data_list1 = [data_dict['language'],data_dict['budget'],data_dict['box_office_collection_worldwide'],data_dict['runtime'],
                  data_dict['release_date'],data_dict['verdict'],data_dict['cbfc_rating']]
    data_list2 = [data_dict['genre'],data_dict['actors'],data_dict['music'],data_dict['director'],data_dict['writer'],data_dict['producer']]
    
    for item in data_list1:
        if item!=item or item in ["",'--','N/A','NA']:
            return True

    for item in data_list2:
        #null check
        if item!=item:
            return True
        else:
            item_cleaned = item.strip('][').split(',')
            if len(item_cleaned)==1 and item_cleaned[0]=='':
                return True

    return False   

def language_preprocess(ln):    
    if ln == 'hi':
        return 0

    elif ln == 'ta':
        return 1

    elif ln == 'te':
        return 2

    elif ln == 'ma':
        return 3

    elif ln == 'kn':
        return 4
        
def genre_preprocess(genre,genre_list):
    keys = genre.strip('][').split(',')
    list_ = [0]*len(genre_list)

    for key in keys:
        key_cleaned = key.strip().strip('\'')
        list_[genre_list.index(key_cleaned)] = 1

    return list_

def release_date_preprocess(date):
    release_month = None
    release_day = None

    if type(date) == str:
        date_list = date.split('-')
        release_day = int(date_list[0])
        release_month = int(date_list[1])
        
    else:
        release_month = date.month
        release_day = date.day
    
    return [release_day,release_month]


def runtime_preprocess(runtime):
    return int(runtime)

def budget_preprocess(budget):
    return int(budget)

def collection_preprocess(collection):
    return int(collection) 

def cbfc_preprocess(rating):
    if rating == 'U':
        return 0

    elif rating == 'UA':
        return 1

    elif rating == 'A':
        return 2

def people_preprocess(peoples_list,popularity_dict):
    popularity_list = []
    
    for people_list in peoples_list:
        score = 0
        keys = people_list.strip('][').split(',')
        
        for key in keys:
            key_cleaned = key.strip().strip('\'')
            if key_cleaned in popularity_dict:
                score+=popularity_dict[key_cleaned]
                
        popularity_list.append(score)

    return popularity_list
            
            
def verdict_preprocess(verdict):
    if verdict in successful_list:
        return 1
    else:
        return 0       

def main():
    # read excel file
    movie_data_files = ['hindi_movies_final.csv','Tamil_movies.csv','telugu_movies_final.csv',
                        'malayalam_movies_final.csv','kannada_movies_final.csv','other_movies_final.xlsx']

    popularity_files = ['Hindi_people.csv','Kannada_people.csv','Malayalam_people.csv','Tamil_people.csv','Telugu_people.csv']
    popularity_dict = {}

    for popularity_file in popularity_files:
        popularity = pd.read_csv(popularity_file)
        for row in range(popularity.shape[0]):
            popularity_dict[popularity.iloc[row,0]] = popularity.iloc[row,1]

    data = []
    frames = []
    genre_set = set()
    
    for file in movie_data_files:
        temp_data = None
        if file.endswith('.csv'):
            temp_data = pd.read_csv(file)
        else:
            temp_data = pd.read_excel(file)

        temp_data = temp_data.iloc[:,:-2]
        #print(temp_data.shape,temp_data.columns)
        frames.append(temp_data)
        
        for genre in temp_data["genre"]:
            if genre!=genre or genre == "":
                continue
            res = genre.strip('][').split(',')
            if len(res) == 0:
                continue
            for item in res:
                genre_set.add(item.strip().strip('\''))
    
    dataframe = pd.concat(frames,ignore_index=True)
    #print(dataframe.columns)
    #dataframe.drop('name', inplace=True, axis=1)

    movies = []
    genre_list = list(genre_set)
    genre_list.sort()
    
    for index in range(dataframe.shape[0]):
        record = dataframe.iloc[index,:]
        if is_null_record(record):
            continue
        data1 = language_preprocess(record['language'])
        data2 = genre_preprocess(record['genre'],genre_list)
        data3 = release_date_preprocess(record['release_date'])
        data4 = runtime_preprocess(record['runtime'])
        data5 = cbfc_preprocess(record['cbfc_rating'])
        data6 = budget_preprocess(record['budget'])
        data7 = collection_preprocess(record['box_office_collection_worldwide'])
        data8 = verdict_preprocess(record['verdict'])
        data9 = data7/data6
        data10 = people_preprocess([record['actors'],record['music'],record['director'],record['writer'],record['producer']],popularity_dict)
                         
        data.append([data1] + data2 + data3 + [data4,data5,data6,data7] + data10 + [data9,data8])
        movies.append(record['name'])
        
    dataarray = np.array(data)
    columns = ['language'] + genre_list + ['release_day','release_month'] + ['runtime','cbfc_rating','budget','box_office_collection_worldwide',
                                                                             'actors','music','director','writer','producer'] + ['roi','verdict']
    file = open('movie_data_1.pkl','wb')
    obj = [columns,movies,dataarray]
    pickle.dump(obj,file)

    file.close()
    
if __name__=='__main__':
    main()
