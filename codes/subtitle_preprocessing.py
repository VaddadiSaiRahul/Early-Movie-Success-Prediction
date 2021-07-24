import os
import srt
import nltk
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import words
import string
import pickle

os.chdir("C:/Users/VADDADISAIRAHUL/Downloads/indian_subtitles_final/")

def subt_to_text(sub_path):
    text = ''
    with open(sub_path, encoding='utf-8-sig') as fd:
        subs = srt.parse(fd)
        for line in subs:
            text += line.content + ' '      
    return text

def preprocessing_text_to_words(text):
    # lower case    
    text_lower_case = text.lower()

    # tokenization
    tokenize_words = nltk.tokenize.word_tokenize(text_lower_case)
    words = [word.strip(string.punctuation) for word in tokenize_words]

    # stop words removal
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_words_1 = [w for w in words if not w in stop_words]

    # non-english words removal
    en_words = set(nltk.corpus.words.words())
    filtered_words_2 = [w for w in filtered_words_1 if w in en_words]

    # lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for index in range(len(filtered_words_2)):
        filtered_words_2[index] = lemmatizer.lemmatize(filtered_words_2[index])

    words_list = [x for x in filtered_words_2 if x != '']
    return words_list

def main():
    indian_films_subtitles_path = ['hindi_subtitles/','tamil_subtitles/','telugu_subtitles/','malayalam_subtitles/','kannada_subtitles/','other_subtitles/']
    preprocess_subtitles_path = ['hindi_subtitles/','other_subtitles/']
    docs = []
    included_movies = []
    subt_words_list = []
    
    i = 0
    for subt_path in indian_films_subtitles_path:
        print(len(included_movies))
        for filename in os.listdir(subt_path):
            try :
                filepath = os.path.join(subt_path,filename)
                srtfilepath = filepath
                srtfilename = filename
                
                if os.path.isdir(filepath):
                    srtfilename = os.listdir(filepath)[0]
                    srtfilepath = os.path.join(filepath,srtfilename)
                    
                text = subt_to_text(srtfilepath)
                words_list = list(set(preprocessing_text_to_words(text)))
                docs.append(TaggedDocument(words = words_list, tags=[str(i)]))
                subt_words_list.append(words_list)
                i+=1

                movie_name = None
                if subt_path in preprocess_subtitles_path:
                    if '_' in srtfilename:
                        temp = ' '.join(srtfilename.split('_'))
                        movie_name = temp.replace('.srt','')
                    else:
                        movie_name = srtfilename.replace('.srt','')    
                else:
                    movie_name = srtfilename.replace('.srt','')
                    
                included_movies.append(movie_name)
                
            except (UnicodeDecodeError, srt.SRTParseError, FileNotFoundError):
                pass
            
    print(len(included_movies))

    file = open('subtitle_train.pkl','wb')
    obj = [included_movies,subt_words_list,docs]
    pickle.dump(obj,file)

    file.close()

if __name__=='__main__':
    main()
